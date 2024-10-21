# *SparseLLM*: Towards Global Pruning of LLMs

This repository contains the code for our **NeurIPS 2024** paper "[*SparseLLM*: Towards Global Pruning for Pre-trained Language Models](https://arxiv.org/abs/2402.17946)".

*WORKING IN PROGRESS*: Our current released code is for preview purposes only and may be subject to numerical instability issues. We are actively working on a more stable version of our method, with an estimated release date around the time our camera-ready paper is out. 

## Updates

- <span style="color:green;">&#x2705;</span> *SparseLLM* code for both **OPT** and **LLaMA** models is now available.
- <span style="color:green;">&#x2705;</span> More model types and functionalities will be added soon.


## Dependencies

This project requires the following core dependencies:

- `Python`: tested on v3.10.14
- `PyTorch`: tested on v2.4.1 with CUDA 12.2 
- `Transformers`: tested on v4.45.1
- `Datasets`: tested on v3.0.1
- `numpy`: tested on v2.1.1
- `pandas`: tested on v2.2.3
- `huggingface_hub`: tested on v0.25.1
- `wandb`: tested on v0.18.2 (for experiment tracking)

## Usage

The scripts directory contains all the bash commands to replicate the main results in our NeurIPS 2024 paper. 

### Example for Pruning OPT:

Below is an example command for pruning the OPT-125M model using SparseLLM, to achieve 70% sparsity.

```
python opt_main.py \
    --model facebook/opt-125m \
    --dataset c4 \
    --sparsity 0.7 \
```

We provide a quick overview of the key arguments:

- `--model`: The identifier for the model on the Hugging Face model hub.
- `--dataset`: The dataset to use for evaluation. We support datasets like `c4`, `wikitext2`, and `ptb`.
- `--sparsity`: The desired sparsity level (percentage of weights to be pruned).

**Remark:** OPT-350M is currently not supported by our method, due to potential numerical stability issue.

### Example for Pruning LLaMA-2:

For **LLaMA-2** models, use the llama_main.py file and specify the model path as `meta-llama/Llama-2-7b-hf`. Here is an example command for pruning LLaMA-2-7B:

```
python llama_main.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset c4 \
    --sparsity 0.7 \
```

### Available Sparsity Methods

We support the following pruning methods for both **OPT** and **LLaMA** models:

- **Unstructured**: Pruning individual weights without any specific pattern.
- **Semi-Structured N:M Sparsity**: For semi-structured pruning, use the following sparsity types:
  - `--sparsity_type 2:4`: Prune 2 out of every 4 weights.
  - `--sparsity_type 4:8`: Prune 4 out of every 8 weights.

```
python opt_main.py \
    --model facebook/opt-125m \
    --dataset c4 \
    --prunen 2 \
    --prunem 4 \
```

Similarly, for **LLaMA-2-7B** semi-structured pruning:

```
python llama_main.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset c4 \
    --prunen 2 \
    --prunem 4 \
```

## Reference

If you find this code useful in your research, please consider citing:

```bibtex
@article{bai2024gradient,
  title={Gradient-Free Adaptive Global Pruning for Pre-trained Language Models},
  author={Bai, Guangji and Li, Yijiang and Ling, Chen and Kim, Kibaek and Zhao, Liang},
  journal={arXiv preprint arXiv:2402.17946},
  year={2024}
}
```
We sincerely appreciate it 😊

## Disclaimer

1. This repository is built upon [SparseGPT](https://arxiv.org/abs/2301.00774) and [Wanda](https://arxiv.org/abs/2306.11695).
2. *SparseLLM* aims to advance the research on improving fully local pruning methods for large language models (LLMs). Due to the iterative alternating optimization nature of *SparseLLM*, its running time will be (roughly number of iteration times) longer than that of one-shot pruning methods such as SparseGPT or Wanda. Additionally, the performance and numerical stability of the alternating optimization process can be sensitive to the initialization of hyperparameters.
3. *SparseLLM* relies on auxiliary variables to achieve subproblem decomposition, which inevitably introduces additional memory overhead. For larger models like LLaMA-2-7B and beyond, we used a smaller calibration data size (e.g., 64 or 32) to ensure the code could run on an A100 40GB GPU. We are actively working on optimizing the GPU memory consumption and improving the efficiency of the code to support larger models and data sizes more effectively.

