import argparse
from model_utils import get_qwen, qwen_sparsellm, llama_eval
from datautils import get_loaders
import torch

def main():
    parser = argparse.ArgumentParser()

    # Arguments parsing
    parser.add_argument("--model", type=str, default='Qwen/Qwen2.5-0.5B', help="Qwen model to load")
    parser.add_argument("--dataset", type=str, choices=["wikitext2", "ptb", "c4"], default="c4", help="Dataset for calibration.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling calibration data.")
    parser.add_argument("--nsamples", type=int, default=32, help="Number of calibration data samples.")
    parser.add_argument("--percdamp", type=float, default=0.01, help="Percent of Hessian diagonal for dampening.")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity.")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument("--blocksize", type=int, default=128, help="Blocksize for adaptive mask selection.")
    parser.add_argument("--gmp", action="store_true", help="Run GMP baseline.")
    parser.add_argument("--wbits", type=int, default=16, help="Quantization bits.")
    parser.add_argument("--minlayer", type=int, default=-1, help="Prune layers with id >= this.")
    parser.add_argument("--maxlayer", type=int, default=1000, help="Prune layers with id < this.")
    parser.add_argument("--prune_only", type=str, default="", help="Prune only layers containing this text.")
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default="", help="Path to save model.")
    parser.add_argument("--true-sequential", action="store_true", help="Run in true sequential mode.")
    parser.add_argument("--log_wandb", action="store_true", help="Log to W&B.")

    args = parser.parse_args()

    model = get_qwen(args)
    model.eval()
    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)

    if (args.sparsity or args.prunen) and not args.gmp:
        qwen_sparsellm(model, dataloader, torch.device('cuda'), args)

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
        llama_eval(model, testloader, torch.device('cuda'), args, dataset)

    if args.save:
        model.save_pretrained(args.save)

if __name__ == '__main__':
    main()
