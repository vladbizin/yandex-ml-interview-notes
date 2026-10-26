# Gradient accumulation

- **Goal**
    - Use a large effective batch when its activations do not fit in GPU memory
- **Main idea**
    - Split the batch into small **microbatches**
    - Run forward and backward for each microbatch
    - Add its gradients to the same buffers
    - Update the weights once, after all microbatches

## Algorithm

- **Terms**
    - $b$: examples per microbatch on one GPU
    - $K$: microbatches per optimizer update - **accumulation steps**
    - $B_{\text{eff}} = bK$:  effective batch size
- **Before the window**
    - Clear gradients
- **For each microbatch**
    - Compute its mean loss and divide it by $K$
    - Run backward
    - Release microbatch's computation graph and activations
    - Keep the accumulated gradients
- **After the window**
    - Clip the accumulated gradient if needed
    - Run the optimizer

## Pitfalls

- **Time cost**
    - Small microbatches can use the GPU less efficiently than one large batch
    - Accumulation enables the desired batch size - it does not guarantee higher throughput

- **Limits**
    - Training-mode BatchNorm uses each microbatch's statistics and updates running statistics each time
    - Losses that compare examples within a batch, such as in-batch contrastive losses, see fewer pairs
    - Dropout draws and floating-point operation order can cause numerical differences

## Sources

- [Hugging Face Accelerate: gradient accumulation and variable-size loss normalization](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)
- [PyTorch: AMP gradient accumulation and clipping](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation)
- [PyTorch: BatchNorm2d statistics during training](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html)
