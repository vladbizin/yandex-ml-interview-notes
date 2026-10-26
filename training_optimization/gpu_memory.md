# GPU memory during training

- **Goal**
    - Estimate how much VRAM training needs on one GPU
- **Main idea**
    - Loading the weights is only the start
    - Training also needs gradients, optimizer states, saved activations, and temporary tensors
    - The model must fit at the **peak** of the training step

## VRAM usage main components

| Component | What it stores | Main size drivers |
| --- | --- | --- |
| **Weights** | Learned parameters | Parameter count, dtype |
| **Gradients** | Loss derivatives for trainable parameters | Trainable parameter count, gradient dtype |
| **Optimizer states** | Update history (Adam keeps first and second gradient moments) | Optimizer, trainable parameter count, and state dtype |
| **Saved activations** | Forward tensors needed to compute gradients | Microbatch size, input size, architecture, and dtype |

## Total VRAM usage

- **Assumptions**
    - Full training: all $P$ parameters are trainable
    - Standard Adam with two FP32 moment tensors

| Storage | FP32 training | Mixed precision with an explicit FP32 master copy |
| --- | ---: | ---: |
| Weights used for computation | $4P$ | $2P$ |
| Separate FP32 master weights for updates | None | $4P$ |
| Gradients (assumed FP32) | $4P$ | $4P$ |
| Adam first and second moments | $8P$ | $8P$ |
| **Subtotal** | **$16P$ bytes** | **$18P$ bytes** |

## VRAM usage during training step

| Stage | Usage |
| --- | --- |
| **Forward** | Save activations |
| **Backward** | Use and then release saved activations - while creating parameter gradients |
| **Optimizer step** | Save momentum tensors (first step only, then update existing) |
| **Clear gradients** | Setting gradients to zero keeps them allocated |

## Large activations reasons

- **Larger input**
- **Attention implementation**
    - Materialized attention scores have shape $[b,h,l,l]$
    - Doubling $l$ makes this tensor four times larger
    - Memory-efficient attention can avoid storing the full score matrix

## Sources

- [Hugging Face: GPU memory usage](https://huggingface.co/docs/transformers/en/model_memory_anatomy)
- [PyTorch: training memory and its timeline](https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html#memory-usage-during-training)
- [PyTorch: Adam and optimizer intermediates](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)
- [PyTorch: CUDA memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management)
