# This file will contain functions related to the model such as loading the model, SparseLLM pruning, and evaluation.

import torch
import torch.nn as nn
from pruning_utils import *
from quant import *
import math
import copy
from transformers import OPTForCausalLM, LlamaForCausalLM

def get_opt(args):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = OPTForCausalLM.from_pretrained(args.model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

def get_llama(args):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def opt_sparsellm(model, dataloader, dev, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
              continue
            gpts[name] = SparseGPT_OPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data, name)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        target_layer_names = ['fc1', 'fc2']

        for name in gpts:
            if name not in target_layer_names:   
                print(i, name)
                print('Pruning ...')
                # Prune the layer
                sparsity = args.sparsity
                gpts[name].fasterprune(
                    sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
                )
                gpts[name].free()

        # Adjust hyperparameters as needed
        alpha = 0.1
        beta = 0.1
        gamma = 0.1

        # Define the number of optimization steps
        opt_epochs = 10

        # Get the inputs and outputs which are constants here
        X_list = gpts['fc1'].batch_inp
        Y_list = gpts['fc2'].batch_out
        X = torch.stack(X_list, dim=0)
        Y = torch.stack(Y_list, dim=0)
        # Reshape to 2D
        X, Y = X.reshape((-1, X.size(-1))).T, Y.reshape((-1, Y.size(-1))).T

        # free memory 
        X_list, Y_list = None, None
        gpts['fc1'].batch_inp.clear()
        gpts['fc2'].batch_out.clear()

        hidden_z_list = gpts['fc1'].batch_out
        z = torch.stack(hidden_z_list, dim=0)
        hidden_z_list = None
        gpts['fc1'].batch_out.clear()
        hidden_p_list = gpts['fc2'].batch_inp
        p = torch.stack(hidden_p_list, dim=0)
        hidden_p_list = None
        gpts['fc2'].batch_inp.clear()

        # Initialize auxiliary variables z and p
        z = z.reshape((-1, z.size(-1))).T.to(dev)
        p = p.reshape((-1, p.size(-1))).T.to(dev)

        torch.cuda.empty_cache()

        # Pre-compute the pinverse of X and cache it to save computational cost
        Xinv = torch.pinverse(X.to(dtype=torch.float32)).half()

        for opt_step in range(opt_epochs):

            ##############
            # optimize W
            ##############

            if opt_step > 0:   # for the first step, no need for updating W

                # Update the weight matrix of fc1
                bias = subset['fc1'].bias.unsqueeze(1).expand(-1, z.size(-1))
                # Calculate the weight matrix
                weight_matrix_1 = torch.matmul(z - bias, Xinv)
                # assign the new parameters to gpts class
                gpts['fc1'].layer.weight.copy_(weight_matrix_1)
                del bias, weight_matrix_1

                # Update the weight matrix of fc2
                if args.sparsity < 0.8:
                    pinv = torch.pinverse(p.to(dtype=torch.float32)).half()
                    bias = subset['fc2'].bias.unsqueeze(1).expand(-1, Y.size(-1))
                    # Calculate the weight matrix
                    weight_matrix_2 = torch.matmul(Y - bias, pinv)
                    # assign the new parameters to gpts class
                    gpts['fc2'].layer.weight.copy_(weight_matrix_2)

                    del bias, weight_matrix_2, pinv
                    torch.cuda.empty_cache()
                else:
                    weight_matrix_2 = copy.deepcopy(gpts['fc2'].layer.weight).to(dtype=torch.float32).requires_grad_(True)
                    bias = subset['fc2'].bias.unsqueeze(1).expand(-1, Y.size(-1))
                    learning_rate = 0.01
                    w_epochs = 10
                    for _ in range(w_epochs):
                        with torch.enable_grad():  
                            y_pred = torch.matmul(weight_matrix_2, p.to(dtype=torch.float32)) + bias.to(dtype=torch.float32)
                            loss = (y_pred - Y.to(dtype=torch.float32)).pow(2).mean()  # L2 loss
                        loss.backward()
                        weight_matrix_2 -= learning_rate * weight_matrix_2.grad                    
                        weight_matrix_2.grad.zero_()
                    weight_matrix_2 = weight_matrix_2.half()
                     # assign the new parameters to gpts class
                    gpts['fc2'].layer.weight.copy_(weight_matrix_2)
               
                    del bias, weight_matrix_2, y_pred
                    torch.cuda.empty_cache()

            ##############
            # prune W
            ##############

            # modify gpts[name].H to be our auxiliary variable
            if opt_step > 0:   # for the first step, no need for updating H    
            
                tmp_H = torch.zeros_like(gpts['fc2'].H)
                tmp_p = p.T.reshape((args.nsamples, -1, p.size(0)))
                tmp_nsamples = 0
                for j in range(args.nsamples):
                    tmp_inp = tmp_p[j].unsqueeze(0)
                    tmp = tmp_inp.shape[0]
                    if isinstance(gpts['fc2'].layer, nn.Linear) or isinstance(gpts['fc2'].layer, transformers.Conv1D):
                        if len(tmp_inp.shape) == 3:
                            tmp_inp = tmp_inp.reshape((-1, tmp_inp.shape[-1]))
                        tmp_inp = tmp_inp.t()
                    tmp_H *= tmp_nsamples / (tmp_nsamples + tmp)
                    tmp_nsamples += tmp
                    tmp_inp = math.sqrt(2 / tmp_nsamples) * tmp_inp.float()
                    tmp_H += tmp_inp.matmul(tmp_inp.t())
                gpts['fc2'].H.copy_(tmp_H)
                del tmp_H, tmp_p
                torch.cuda.empty_cache()

            for name in target_layer_names:
                print(i, name)
                print('Pruning ...')
                sparsity = args.sparsity
                gpts[name].fasterprune(
                    sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
                )

            ##############
            # optimize p
            ##############

            # Activation inverse
            next_weight = subset['fc2'].weight
            m1 = beta * torch.matmul(next_weight.T, next_weight)
            m2 = gamma * torch.eye(m1.shape[0], device=m1.device)
            av = torch.inverse(m1 + m2).to(dtype=torch.float16)

            del m1, m2
            torch.cuda.empty_cache()

            # Calculate ReLU
            layer_nl_output = nn.functional.relu(z)

            # Activation formulate
            bias = subset['fc2'].bias.unsqueeze(1).expand(-1, Y.size(-1))
            m3 = beta * torch.matmul(next_weight.T, Y - bias)
            m4 = gamma * layer_nl_output
            af = m3 + m4

            p = torch.matmul(av, af)

            del layer_nl_output, next_weight, av, m3, m4, af, bias
            torch.cuda.empty_cache()

            ##############
            # optimize z
            ##############

            w = subset['fc1'].weight
            bias = subset['fc1'].bias.unsqueeze(1).expand(-1, z.size(-1))
            m = torch.matmul(w, X) + bias
            sol1 = (gamma * p + alpha * m) / (gamma + alpha)
            sol2 = m
            del w, bias
            torch.cuda.empty_cache()

            z1 = torch.zeros_like(p)
            z2 = torch.zeros_like(p)

            chunk_size = 500  # Choose an appropriate size based on your memory constraints
            # Assuming the first dimension is the one to be chunked
            for k in range(0, sol1.size(0), chunk_size):
                chunk = slice(k, k + chunk_size)
                
                # Apply the condition and assignment for the chunk
                z1_chunk = z1[chunk]
                sol1_chunk = sol1[chunk]
                z1_chunk[sol1_chunk >= 0.] = sol1_chunk[sol1_chunk >= 0.]
                z1[chunk] = z1_chunk

                z2_chunk = z2[chunk]
                sol2_chunk = sol2[chunk]
                z2_chunk[sol2_chunk <= 0.] = sol2_chunk[sol2_chunk <= 0.]
                z2[chunk] = z2_chunk

            del z1_chunk, z2_chunk, sol1_chunk, sol2_chunk, sol1, sol2
            torch.cuda.empty_cache()

            for k in range(0, z1.size(0), chunk_size):
                chunk = slice(k, k + chunk_size)
                
                # Compute fz_1 and fz_2 for the current chunk
                fz_1_chunk = gamma * torch.square(p[chunk] - nn.functional.relu(z1[chunk])) + alpha * torch.square(z1[chunk] - m[chunk])
                fz_2_chunk = gamma * torch.square(p[chunk] - nn.functional.relu(z2[chunk])) + alpha * torch.square(z2[chunk] - m[chunk])

                # Determine indices for z1 and z2 for the current chunk
                index_z1_chunk = fz_1_chunk <= fz_2_chunk
                index_z2_chunk = fz_2_chunk < fz_1_chunk

                # Update z for the current chunk
                z[chunk][index_z1_chunk] = z1[chunk][index_z1_chunk]
                z[chunk][index_z2_chunk] = z2[chunk][index_z2_chunk]

            # Clear memory if necessary
            del fz_1_chunk, fz_2_chunk, index_z1_chunk, index_z2_chunk, z1, z2, m, chunk
            torch.cuda.empty_cache()

        for name in target_layer_names:
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def llama_sparsellm(model, dataloader, dev, args):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
                    continue
                gpts[name] = SparseGPT_LlaMA(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data, name)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            target_layer_names = ["mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"]

            for name in subset:
                if name not in target_layer_names:   
                    print(i, name)
                    print("Pruning ...")
                    sparsity = args.sparsity
                    gpts[name].fasterprune(
                        sparsity,
                        prunen=args.prunen,
                        prunem=args.prunem,
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                    )
                    gpts[name].free()

            # Adjust hyperparameters as needed
            alpha = 5.0
            beta = 5.0
            gamma = 5.0

            # Define the number of global pruning epochs
            opt_epochs = 8  # This might need to be adjusted

            # Get the inputs and outputs which are constants here
            X_list = gpts['mlp.up_proj'].batch_inp
            Y_list = gpts['mlp.down_proj'].batch_out
            X = torch.stack(X_list, dim=0)
            Y = torch.stack(Y_list, dim=0)
            # Reshape to 2D
            X, Y = X.reshape((-1, X.size(-1))).T, Y.reshape((-1, Y.size(-1))).T

            # free memory 
            X_list, Y_list = None, None
            gpts['mlp.up_proj'].batch_inp.clear()
            gpts['mlp.down_proj'].batch_out.clear()

            # Get the hidden variables and their initialization
            # z: output of 'mlp.up_proj'
            hidden_z_list = gpts['mlp.up_proj'].batch_out
            z = torch.stack(hidden_z_list, dim=0)
            hidden_z_list = None
            gpts['mlp.up_proj'].batch_out.clear()
            # p: input of 'mlp.down_proj'
            hidden_p_list = gpts['mlp.down_proj'].batch_inp
            p = torch.stack(hidden_p_list, dim=0)
            hidden_p_list = None
            gpts['mlp.down_proj'].batch_inp.clear()
            # s: output of 'mlp.gate_proj'
            hidden_s_list = gpts['mlp.gate_proj'].batch_out
            s = torch.stack(hidden_s_list, dim=0)
            hidden_s_list = None
            gpts['mlp.gate_proj'].batch_out.clear()

            # Reshape auxiliary variables
            z = z.reshape((-1, z.size(-1))).T.to(dev)
            p = p.reshape((-1, p.size(-1))).T.to(dev)
            s = s.reshape((-1, s.size(-1))).T.to(dev)

            torch.cuda.empty_cache()

            # Pre-compute the pinverse of X and cache it to save computational cost
            Xinv = torch.pinverse(X.to(dtype=torch.float32)).half()

            # list to store training losses
            training_loss = {'Y_p_loss': [], 'p_z_loss': [], 'z_X_loss': [], 'train_loss': []}

            for opt_step in range(opt_epochs):

                ##############
                # optimize W
                ##############

                if opt_step > 0:   # for the first step, no need for updating W

                    # Update the weight matrix of mlp.up_project
                    # Calculate the weight matrix
                    weight_matrix_1 = torch.matmul(z, Xinv)
                    # assign the new parameters to gpts class
                    gpts['mlp.up_proj'].layer.weight.copy_(weight_matrix_1)
                    del weight_matrix_1

                    # Update the weight matrix of mlp.down_proj
                    pinv = torch.pinverse(p.to(dtype=torch.float32)).half()
                    # Calculate the weight matrix
                    weight_matrix_2 = torch.matmul(Y, pinv)
                    # assign the new parameters to gpts class
                    gpts['mlp.down_proj'].layer.weight.copy_(weight_matrix_2)
                    del weight_matrix_2, pinv

                    # Update the weight matrix of mlp.gate_project
                    # Calculate the weight matrix
                    weight_matrix_3 = torch.matmul(s, Xinv)
                    # assign the new parameters to gpts class
                    gpts['mlp.gate_proj'].layer.weight.copy_(weight_matrix_3)
                    del weight_matrix_3

                    torch.cuda.empty_cache()

                ##############
                # prune W
                ##############

                # modify gpts[name].H to be our auxiliary variable
                if opt_step > 0:   # for the first step, no need for updating H    
                
                    tmp_H = torch.zeros_like(gpts['mlp.down_proj'].H)
                    tmp_p = p.T.reshape((args.nsamples, -1, p.size(0)))
                    tmp_nsamples = 0
                    for j in range(args.nsamples):
                        tmp_inp = tmp_p[j].unsqueeze(0)
                        tmp = tmp_inp.shape[0]
                        if isinstance(gpts['mlp.down_proj'].layer, nn.Linear) or isinstance(gpts['mlp.down_proj'].layer, transformers.Conv1D):
                            if len(tmp_inp.shape) == 3:
                                tmp_inp = tmp_inp.reshape((-1, tmp_inp.shape[-1]))
                            tmp_inp = tmp_inp.t()
                        tmp_H *= tmp_nsamples / (tmp_nsamples + tmp)
                        tmp_nsamples += tmp
                        tmp_inp = math.sqrt(2 / tmp_nsamples) * tmp_inp.float()
                        tmp_H += tmp_inp.matmul(tmp_inp.t())
                    gpts['mlp.down_proj'].H.copy_(tmp_H)
                    del tmp_H, tmp_p
                    torch.cuda.empty_cache()

                for name in target_layer_names:
                    print(i, name)
                    print('Pruning ...')
                    sparsity = args.sparsity
                    gpts[name].fasterprune(
                        sparsity,
                        prunen=args.prunen,
                        prunem=args.prunem,
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                    )

                ##############
                # optimize p
                ##############

                # Activation inverse
                next_weight = subset['mlp.down_proj'].weight
                m1 = beta * torch.matmul(next_weight.T, next_weight)
                m2 = gamma * torch.eye(m1.shape[0], device=m1.device)
                av = torch.inverse(m1 + m2).to(dtype=torch.float16)

                del m1, m2
                torch.cuda.empty_cache()

                # Calculate SwiGLU output
                layer_nl_output = nn.functional.silu(s) * z

                # Activation formulate
                m3 = beta * torch.matmul(next_weight.T, Y)
                m4 = gamma * layer_nl_output
                af = m3 + m4

                p = torch.matmul(av, af)

                del layer_nl_output, next_weight, av, m3, m4, af
                torch.cuda.empty_cache()

                ##############
                # optimize z
                ##############

                w = subset['mlp.up_proj'].weight
                m = torch.matmul(w, X)
                swish = nn.functional.silu(s)
                z = (m + swish * p) / (swish ** 2 + 1)    

                del w, m, swish
                torch.cuda.empty_cache()

                ##############
                # optimize s
                ##############

                w = subset['mlp.gate_proj'].weight
                # convert the layer's weight tensor to float32 and enable grad
                w = w.to(dtype=torch.float32).requires_grad_(True)

                s_update_epochs = 2
                s_learning_rate = 0.01
                for _ in range(s_update_epochs):

                    batch_size = 1000  # Choose an appropriate batch size based on your memory constraints
                    # s: [hidden_d, n_samples]
                    for k in range(0, s.size(-1), batch_size):
                        chunk = slice(k, k + batch_size)

                        # get the "mini-batch" for each tensor and turn on autograd
                        X_batch = X[:,chunk].to(dtype=torch.float32).requires_grad_(True)
                        z_batch = z[:,chunk].to(dtype=torch.float32).requires_grad_(True)
                        p_batch = p[:,chunk].to(dtype=torch.float32).requires_grad_(True)
                        s_batch = s[:,chunk].to(dtype=torch.float32).requires_grad_(True)

                        with torch.enable_grad():   # temporarily turn on the Pytorch computational graph functionality

                            loss_s = alpha * torch.norm(s_batch - torch.matmul(w, X_batch))**2
                            loss_s += gamma * torch.norm(p_batch - nn.functional.silu(s_batch) * z_batch)**2

                        loss_s.backward()
                        s_batch -= s_learning_rate * s_batch.grad                    
                        s_batch.grad.zero_()
                        s[:,chunk] = s_batch.detach().to(dtype=torch.float16)

                s_batch, X_batch, z_batch, p_batch, w = s_batch.detach(), X_batch.detach(), z_batch.detach(), p_batch.detach(), w.detach()
                del w, loss_s, s_batch, X_batch, z_batch, p_batch
                torch.cuda.empty_cache()

                # compute and save the training loss after each epoch
                tmp_training_loss = nn.functional.mse_loss(torch.matmul(subset['mlp.down_proj'].weight, 
                                                                        nn.functional.silu(torch.matmul(subset['mlp.gate_proj'].weight, X)) 
                                                                        * torch.matmul(subset['mlp.up_proj'].weight, X)), Y)
                training_loss['train_loss'].append(tmp_training_loss.item())

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def opt_eval(model, testenc, dev, args, dataset: str):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache



@torch.no_grad()
def llama_eval(model, testenc, dev, args, dataset: str):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][
                    int(W.numel() * args.sparsity)
                ]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache