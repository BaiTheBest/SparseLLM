# Adaptive Global Pruning (AdaGP)

WORKING IN PROGRESS.

This repository contains code for the preprint paper "Gradient-Free Adaptive Global Pruning for Pre-trained Language Models". This repo is built upon that of the paper [SparseGPT: Massive Language Models Can be Accurately Pruned in One-shot](https://arxiv.org/abs/2301.00774).

Specifically, we currently only released the version of AdaGP for pruning OPT-125M, and the complete version will be released soon. Please stay tuned! 

## Dependencies

* `Python`: tested on v3.8
* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2
* `datasets`: tested on v1.17.0

## Usage

Here are some sample commands to run baselines and sparsification on OPT-125M model, followed by perplexity evaluations on raw-WikiText2, PTB and C4.
Our code has been tested on a single NVIDIA A100 40GB.

```
# Prune to 70\% uniform sparsity with AdaGP
python adagp.py --model facebook/opt-125m --dataset c4 --sparsity 0.7

# Prune to 70\% uniform sparsity with SparseGPT
python opt.py facebook/opt-125m c4 --sparsity 0.7
```

## Reference

If you find this code useful in your research, please consider citing:

        @article{bai2024gradient,
        title={Gradient-Free Adaptive Global Pruning for Pre-trained Language Models},
        author={Bai, Guangji and Li, Yijiang and Ling, Chen and Kim, Kibaek and Zhao, Liang},
        journal={arXiv preprint arXiv:2402.17946},
        year={2024}
        }
