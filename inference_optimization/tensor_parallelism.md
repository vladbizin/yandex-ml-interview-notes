# Tensor parallelism for inference

- **Goal**
    - Fit a model across GPUs and reduce the work and weight reads on each GPU
- **Main idea**
    - Split large weight matrices inside a layer
    - Keep intermediate features split
    - Communicate when partial results must be combined

## Parallelism Options

- **Convention:** write the linear layer as $Y = XW$; omit biases
    - $X$: input token features. $W$: weights. $Y$: output token features
    - Row/column names below refer to $W$ in this convention
- **Column-parallel: split output features**
    - Split $W$ into column blocks $W_i$. Each GPU has the full $X$
    - Each GPU computes $Y_i = XW_i$. These outputs are different slices of $Y$
    - An **all-gather** concatenates the slices if the next operation needs full $Y$
- **Row-parallel: split input features**
    - Split $X$ into feature slices $X_i$ and $W$ into matching row blocks $W_i$
    - Each GPU computes $P_i = X_iW_i$. Each partial result has the full output shape
    - An **all-reduce sum** computes $Y = \sum_i P_i$ and gives it to every GPU
- **Useful pairing: column → local operation → row**
    - The column split produces exactly the feature slices that the row split needs
    - Keep them local. No intermediate all-gather is needed

## Example

![No TP](https://cdn-uploads.huggingface.co/production/uploads/631ce4b244503b72277fc89f/EN0Sf-XN6wZBs3BXFnCN4.png)

![Column-Parallel](https://cdn-uploads.huggingface.co/production/uploads/631ce4b244503b72277fc89f/mmXOetPIx98yNLSHxb4rd.png)

![Row-Parallel](https://cdn-uploads.huggingface.co/production/uploads/631ce4b244503b72277fc89f/rbCzHY4a9Ab89xZ80sNDl.png)

## Attention

- **Q/K/V projections**
    - Split their output columns by attention head
    - Each GPU computes its assigned heads and stores their KV cache
- **Output projection**
    - Split its input rows to match the local heads, then all-reduce the partial outputs
- **Communication count**
    - This layout uses two all-reduces per transformer block per forward pass
        - after attention
        - after the FFN
- **Constraints**
    - An even split needs compatible head counts and FFN widths

![Attention TP](https://cdn-uploads.huggingface.co/production/uploads/631ce4b244503b72277fc89f/ZtdDlW4HW8CuNmj0Q9LDC.png)

## Inference

- **Prefill**
    - Many prompt tokens provide large matrix multiplications
    - TP divides their compute
- **Decode**
    - Small batches often spend much of their time reading weights and KV cache
    - TP divides weight reads across GPU memory systems, but adds collective latency at each step
- **Scaling**
    - More GPUs mean smaller local work and frequent synchronization
    - Fast links such as NVLink help

## Sources

- [Quentin Gallouédec: Tensor Parallelism in Transformers — 5 Minutes to Understand](https://huggingface.co/blog/qgallouedec/tp)
- [Hugging Face: Ultra-Scale Playbook — Tensor Parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)
- [NVIDIA: Mastering LLM Techniques — Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization#tensor_parallelism)
