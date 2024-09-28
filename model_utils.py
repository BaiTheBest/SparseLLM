# This file will contain functions related to the model such as loading the model, SparseLLM pruning, and evaluation.

import torch
import torch.nn as nn
from pruning_utils import *
from quant import *
import math
from transformers import OPTForCausalLM

def get_opt(args):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = OPTForCausalLM.from_pretrained(args.model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
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
        alpha = 5.0
        beta = 5.0
        gamma = 5.0

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
                pinv = torch.pinverse(p.to(dtype=torch.float32)).half()
                bias = subset['fc2'].bias.unsqueeze(1).expand(-1, Y.size(-1))
                # Calculate the weight matrix
                weight_matrix_2 = torch.matmul(Y - bias, pinv)
                # assign the new parameters to gpts class
                gpts['fc2'].layer.weight.copy_(weight_matrix_2)

                del bias, weight_matrix_2, pinv
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
