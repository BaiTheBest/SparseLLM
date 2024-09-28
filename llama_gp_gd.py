import time

import torch
import torch.nn as nn

from llama_gp_utils import *
from modelutils import *
from quant import *

# Additional denepencies than SparseGPT's code
import torch.optim as optim
import copy
import numpy as np
from matplotlib import pyplot as plt

import os
os.environ['HF_HOME'] = '/lcrc/project/NEXTGENOPT/yijiang/cache'

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False

# self-defined function to monitor GPU memory consupmtion
def print_memory_usage(device):
    print(f'Current memory allocated on {device}: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB')
    print(f'Peak memory allocated on {device}: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB')
    torch.cuda.reset_peak_memory_stats(device)  # Reset peak memory stats after printing


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, 
                                             torch_dtype='auto', 
                                             token='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', 
                                             cache_dir='/lcrc/project/NEXTGENOPT/yijiang/cache')
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev):
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

    quantizers = {}

    # Control which layers to prune
    prune_layer_index = [1, 2]
    prune_layer_index = [x - 1 for x in prune_layer_index]
    print('Decoder layers to prune:', prune_layer_index)

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
                gpts[name] = SparseGPT(subset[name])
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

            if i in prune_layer_index:
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

            # Initialize the hyperparameters for the AdaGP method
            # Adjust hyperparameters as needed
            alpha = 5.0
            beta = 5.0
            gamma = 5.0

            # Define the number of global pruning epochs
            adagp_epochs = 0  # This might need to be adjusted
            if i in prune_layer_index:
                adagp_epochs = 8

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

            for adagp_step in range(adagp_epochs):

                ##############
                # optimize W
                ##############

                if adagp_step > 0:   # for the first step, no need for updating W

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
                if adagp_step > 0:   # for the first step, no need for updating H    
                
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

        # After training, save the plot of L2 norms to a file
        if adagp_epochs > 1:
            if i in prune_layer_index:
                plt.figure(figsize=(15, 6))
                for key, values in training_loss.items():
                    plt.plot(values, label=key)
                plt.xlabel('Training Steps')
                plt.ylabel('L2 Norm')
                plt.title('Training Objective Curves')
                plt.legend()
                plt.savefig(f'training_objective_layer_{i+1}_{args.model.split("/")[-1]}_{100 * args.sparsity}_{args.nsamples}.png')  # Saves the plot as a PNG file

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
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
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='meta-llama/Llama-2-7b-hf', help="LlaMA model to load")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        default="c4",
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=32, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only",
        type=str,
        default="",
        help="Prune only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default="", help="Path to saved model.")
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        llama_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'down_proj' in n:
                break
        print(time.time() - tick)

    for dataset in ["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print("Dataset:", dataset)
        llama_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)
