# MoE inference optimization

- **Starting point**
    - An **expert** is a feed-forward network with its own weights
    - At each MoE layer, a **router** chooses a few experts for each token
    - Their outputs are combined with the router's weights
    - Only selected experts compute for that token
    - All expert weights must remain accessible

## Possible problems

1. **The model does not fit on one GPU:** too many stored expert weights.
2. **The GPU does many small tasks:** tokens go to different experts.
3. **Generation is slow even with good kernels:** too few tokens reuse each expert's weights.
4. **Some GPUs wait for others:** expert workloads are uneven.
5. **Multiple GPUs spend too much time exchanging data:** tokens need experts on other devices.

## 1. The model does not fit on one GPU

- **Problem**
    - A model activates only a few experts per token, yet its weights exceed GPU memory
- **Why it happens**
    - “Active parameters” describe the weights used for one token
    - The next token can select different experts. Unselected experts cannot simply be deleted
    - Weight storage therefore depends on **all experts**, not only the selected ones
- **Solution**
    - **Quantization:** store weights with fewer bits per value
    - **Expert parallelism (EP):** keep different experts on different GPUs
    - If GPU memory is still insufficient, **offload** some weights to CPU memory and fetch them when needed
- **Cost to remember**
    - Quantization can affect quality and needs suitable kernels
    - EP adds token transfers between GPUs; offloading adds weight transfers from CPU memory

## 2. The GPU does many small tasks

- **Problem**
    - The model fits, but the GPU spends too much time starting small operations
- **Why it happens**
    - Different tokens select different experts
    - A simple implementation handles each token–expert pair separately
    - Many small matrix operations use the GPU less efficiently than larger operations
- **Solution**
    - **Group tokens by expert** before running the expert networks
    - Use **grouped matrix multiplication** to run these expert groups efficiently
    - Use **fused kernels** to combine stages and avoid extra launches or intermediate memory writes
- **Cost to remember**
    - Grouping also takes time
    - A backend that helps large inputs may be slower for very small inputs

## 3. Generation has too few tokens to reuse expert weights

- **Problem**
    - Grouping is implemented, but generation for one request is still slow
- **Why it happens**
    - **Decode** processes one new input token per active request at each step
    - With one request, each selected expert receives only one token vector
    - Reading a large expert's weights can cost more time than the arithmetic on that vector
    - Grouping cannot create more work when there are no other tokens to group
- **Solution**
    - **Batch active requests together**, using continuous batching as requests arrive and finish
    - For small decode batches, use kernels designed for low overhead
    - Quantization can also reduce the weight bytes read
- **Why it works**
    - Tokens from different requests can select the same expert
    - That expert processes them together and reuses its weights
- **Cost to remember**
    - Waiting to build a larger batch increases latency

## 4. Some GPUs wait while another GPU is busy

- **Problem**
    - Experts are spread evenly across GPUs, but one GPU finishes much later
- **Why it happens**
    - The router chooses experts from token content, not from the number of free GPUs
    - Some experts receive more tokens than others
    - The next stage may need results from every GPU - one busy device delays progress
- **Solution**
    - Measure how much work each expert receives
    - **Rebalance placement:** spread busy experts across GPUs
    - If one expert is especially busy, **replicate** it and divide its assignments between identical copies

## 5. Token transfers erase the benefit of multiple GPUs

- **Problem**
    - Adding GPUs makes the weights fit, but does not make inference faster
- **Why it happens**
    - With EP, a token may need an expert stored on another GPU
    - Its vector must travel to that GPU, and the expert's output must travel back
    - When expert computation is short, transfer and synchronization can dominate
- **Solution**
    - Connect cooperating GPUs through fast links and use an efficient token-exchange backend
    - Compare smaller EP groups when memory permits
    - With enough work, **overlap communication and computation** using smaller batches of tokens
- **Why it works**
    - Faster links reduce transfer time; an efficient backend reduces exchange overhead
    - A smaller EP group can avoid slow links and reduce coordination
    - Overlap lets the GPU compute one batch while another batch's vectors travel
- **Cost to remember**
    - Overlap needs independent work - the same token must wait for its own transfer before its expert can run
    - Smaller batches can make matrix operations less efficient

## Sources

- [Hugging Face: Expert parallelism](https://huggingface.co/docs/transformers/en/expert_parallelism)
- [Hugging Face: Experts backends](https://huggingface.co/docs/transformers/en/experts_interface)
- [Mixture of Experts Explained](https://huggingface.co/blog/moe)
