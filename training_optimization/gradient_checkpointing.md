# Gradient checkpointing

- **Goal**
    - Reduce saved activation memory during training
- **Main idea**
    - Also called **activation checkpointing** or **activation recomputation**
    - Keep selected boundary tensors instead of all tensors needed for backward
    - Recompute missing tensors during backward
    - Trade extra compute for lower peak GPU memory

## Algorithm

- **Without checkpointing**
    - Backward needs tensors from the forward pass to compute gradients
    - Autograd saves these tensors until backward uses them
- **Forward with checkpointing**
    - Split the model into regions, such as groups of transformer blocks
    - Keep each checkpointed region's inputs
    - Run the region without keeping its internal tensors for backward
    - Pass its output to the next region as usual
- **Backward with checkpointing**
    - Visit regions in reverse order
    - Rerun forward operations from the saved inputs to rebuild needed tensors
    - Compute gradients, then release those tensors
    - Continue backward into the previous region

## Tradeoffs

- **Compute cost**
    - Backward now includes repeated forward work
- **Region size**
    - Small regions keep more boundary tensors
    - Large regions can need more temporary memory during recomputation
    - Selective checkpointing can keep expensive operation outputs and recompute cheaper ones

## Pitfalls

- **Same computation**
    - Recomputed tensors must match the original forward computation
- **Dropout**
    - Recompute with the same random choices

## Sources

- [PyTorch: activation checkpointing and selective recomputation](https://pytorch.org/blog/activation-checkpointing-techniques/)
- [PyTorch: checkpoint API, random state, and correctness limits](https://docs.pytorch.org/docs/2.14/checkpoint.html)

