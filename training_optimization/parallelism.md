# Parallelism methods for training

- **Goal**
    - Train faster, or fit training across several GPUs
- **Main idea**
    - Split **examples** with data parallelism
    - Split **stored training state** with ZeRO / FSDP
    - Split **layers** with pipeline parallelism
    - Split **operations inside a layer** with tensor parallelism
    - Combine these methods when one split is not enough

## Comparison table

| Method | Work on each GPU | Main communication | Main limit |
| --- | --- | --- | --- |
| **Data parallelism (DDP)** | Full model, different examples | Synchronize gradients | Full training state must fit per GPU |
| **ZeRO / FSDP** | Different examples, some training state stored as shards | Exchange state and reduce gradients | Temporary full parameters and activations still need memory |
| **Pipeline parallelism (PP)** | A group of layers, different microbatches | Send activations forward and their gradients backward | Idle time and uneven stage costs |
| **Tensor parallelism (TP)** | Part of a layer operation on the same examples | Combine partial results inside layers | Frequent communication |

- **Terms**
    - A **rank** is a process in a distributed job; usually one process per GPU
    - A **shard** is one part of a tensor or training state
    - **Model parallelism** is the broad idea of splitting a model; PP and TP are two forms

## Data parallelism

- **Mechanism**
    - Each rank keeps full weights, gradients, and optimizer states
    - Each rank runs forward and backward on its own microbatch
    - An **all-reduce** combines gradients and gives the result to every rank
    - Each rank applies the same optimizer update; weights stay synchronized
- **Distributed data parallelism**
    - Same but supports distributed training across **multiple machines** with multiple GPUs

## ZeRO / FSDP

- **Mechanism**
    - Ranks still process different examples, as in data parallelism
    - Store one shard of selected state per rank instead of a full copy
    - Each parameter has an owner rank; its update uses gradients from **all ranks' batches**

| Stage | Sharded across data parallel ranks | Still replicated | Memory reduction | Communication volume |
| --- | --- | --- | --- | --- |
| **ZeRO-1** | Optimizer states | Compute weights, gradients | 4x | Same as data parallelism |
| **ZeRO-2** | Optimizer states, gradients | Compute weights | 8x | Same as data parallelism |
| **ZeRO-3** | Optimizer states, gradients, compute weights | Full parameters only when needed for computation | `parallelism degree` | +50% |

- **Stage 1**
    - Each rank has all weights and computes gradients for all parameters on its own batch
    - Combine gradients across ranks; each owner uses the averaged gradients for its parameters
    - The owner updates its Adam history and its parameter shard
    - Share updated weight shards so every rank has the full model again
- **Stage 2**
    - Backward still computes local gradients for all parameters
    - As chunks become ready, **reduce-scatter** combines gradients and leaves each owner its averaged shard
    - Release other local gradient chunks instead of keeping full gradient buffers
    - Update owned parameters, then **all-gather** updated weight shards to restore full weights on every rank
- **Stage 3**
    - Before a layer or block runs, **all-gather** its weight shards into full weights on each rank
    - Each rank computes with those weights on its own examples
    - Release gathered weights after use; gather again for backward when needed
    - Reduce-scatter gradients and update owned weights; weights can stay sharded until needed again
    - Peak memory must still fit gathered weights, activations, and temporary tensors

## Pipeline parallelism

- Distributes a model across multiple GPUs (Typically layers or model blocks aredistributed)
- **Mechanism**    
    - Split models into **stages** and the local batch into **microbatches**
    - While one stage processes a microbatch, another stage can process a different one
    - Send activations to the next stage; send activation gradients back during backward
    - Accumulate parameter gradients and update once after the microbatches

## Tensor parallelism

- **Mechanism**
    - GPUs cooperate on the same microbatch inside a layer
    - Split weight matrices and compute partial outputs in parallel
    - Exchange results during forward and backward

## Hybrid parallelism

### 2D: Data parallelism and pipeline parallelism

* Each data parallel rank treats the process as if there were only one GPU instead
* GPUs withing data parallel rank can offload micro-batches of data and reduce idle time

![2d hybrid parallelism](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero-dp-pp.png)

### 3D: ZeRO data parallelism, pipeline parallelism, and model parallelism

- **Memory efficiency** achieved by
    - Splitting the model across GPUs
    - Dividing it into stages to create a pipeline
    - Allows GPUs to work in parallel on micro-batches of data
- **Compute efficiency**
    - Enabled by ZeRO data parallelism
    - Each GPU only stores a slice of the model, optimizer, and activations
    - Allows higher communication bandwidth between data parallel nodes

![3d hybrid parallelism](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-deepspeed-3d.png)

## Choosing a method

| Scenario | Method | 
| --- | --- | 
| Full training fits per GPU | DDP for more throughput |
| Model state is too large | ZeRO / FSDP or PP |
| One large operation is too large | TP | 

## Sources

- [Hugging Face: Parallelism methods](https://huggingface.co/docs/transformers/en/perf_train_gpu_many)
- [Hugging Face: Model Parallelism, v4.13.0](https://huggingface.co/docs/transformers/v4.13.0/en/parallelism)
- [DeepSpeed: ZeRO stages and offload](https://www.deepspeed.ai/tutorials/zero/)
- [ZeRO & DeepSpeed](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
