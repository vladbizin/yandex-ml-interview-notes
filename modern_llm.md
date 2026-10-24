# Modern LLM architectures

- **Main Ideas**
    - **RoPE:** put position into queries and keys
    - **RMSNorm + gated FFNs:** change the computation inside each block
    - **GQA + local attention:** reduce attention memory and work
    - **MoE:** store many feed-forward networks, but run only a few per token

## 1. Backbone

Stays the same

- Decoder-only, autoregressive
- Each block
    - Attention
    - Feed-forward network (FFN)
- Residual stream

![A side-by-side comparison between gpt-oss-20b and GPT-2 XL 1.5B](https://substackcdn.com/image/fetch/$s_!-dhg!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa8fb9d66-b650-441c-a2da-a0931fddb068_1425x769.png)

## 2. Position and Normalization

### Learned positions → RoPE

- **GPT-2**
    - add two vectors at the input: token embedding + learned position embedding
- **RoPE — rotary position embedding**
    - Rotate pairs of query and key coordinates by position-dependent angles
    - Different coordinate pairs rotate at different rates
    - The query–key dot product then depends on their relative position

### LayerNorm → RMSNorm

- **LayerNorm**
    - Subtract the feature mean
    - Divide by the feature standard deviation
    - Apply learned scale and bias
- **RMSNorm**
    - Divide by the root mean square, apply learned scale
    - Does not subtract the mean
    - RMSNorm removes some operations

Changing LayerNorm to RMSNorm does not change where residual additions occur

## 4. Attention

### MHA → GQA

- **MHA**
    - Each query head has its own K/V head
- **GQA**
    - A group of query heads shares one K/V head
    - The queries stay different - can produce different attention weights over the shared keys

GQA trades independent K/V representations for lower cache memory and bandwidth demand

### Full attention → alternating local and full

- **Full causal layer**
    - Position $t$ can read positions $0 \dots t$
- **Local causal layer**
    - With window size $W$, it reads $\max(0, t-W+1) \dots t$
- **Alternating**
    - Local layers handle nearby context with fewer attention pairs
    - Full layers provide direct access to the earlier prefix
    - Earlier information can also reach local layers through contextualized hidden states
- **Note**
    - Keeping full layers means the whole model still has a quadratic attention term
    - A serving implementation can bound the local layers' KV cache. Full layers still need the full prefix

### Learned attention sinks

- **Ordinary softmax**
    - The weights over allowed tokens must sum to 1
- **gpt-oss**
    - Add one learned sink score per head to the softmax inputs
    - Discard its output weight before combining real values. This acts like an extra zero-valued choice
    - Real-token weights can now sum to less than 1, so the head can reduce its contribution
    - This is not an extra text token or a token stored in the KV cache

## 5. Feed-forward layers

### Dense GELU FFN → gated SwiGLU FFN

- **Classic FFN**
    - Project up → GELU → project down
- **Standard SwiGLU**
    - Make two projections, multiply their outputs, then project down
    - Apply SiLU / Swish to one branch: $\text{SiLU}(a) = a  \sigma(a)$
    - In short: $\text{FFN}(x) = [\text{SiLU}(xW_g) \odot (xW_u)]W_d$
    - $W_g$ and $W_u$ map model width to intermediate width; $W_d$ maps back
    - $\odot$ means elementwise multiplication. One branch controls the other branch's features

![A comparison between Swish and GELU and their gated counterparts, SwiGLU and GEGLU](https://substackcdn.com/image/fetch/$s_!8gzt!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F10a8befb-b407-45a4-b38b-486c3d7a65d6_1005x844.png)

### One FFN → mixture of experts (MoE)

- **Expert**
    - An FFN with its own weights
    - Sits inside a transformer block - not separate complete language models
    - Each block has its own expert bank and router
- **Router**
    - Score all experts from each token's normalized hidden vector
    - Choose the top `k` experts for that token
    - gpt-oss applies softmax **over the selected scores**, then sums the expert outputs with those weights
    - Routing happens again at every block
    - The same token can take different routes at different depths
- **Sparse activation**
    - Only selected experts run for a token
    - Each selected expert still uses dense matrix multiplication
    - Attention and the other shared parts still run
    - Experts can develop patterns of specialization

![The feed forward module is replaced by a Mixture-of-Expert (MoE) module](https://substackcdn.com/image/fetch/$s_!SYqb!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffa3367a4-914a-49e0-8c94-016969397ab3_1307x640.png)

## 6. Comparison Table

| Component | Classic GPT-2 | GOT-OSS | Main point |
| --- | --- | --- | --- |
| Position | Learned absolute embeddings added to token embeddings | RoPE on queries and keys; YaRN context extension | Position enters attention differently |
| Normalization | LayerNorm | RMSNorm | Normalize scale without subtracting the mean |
| Attention heads | Multi-head attention (MHA) | Grouped-query attention (GQA) | Share K/V across query heads |
| Context access | Full causal attention in every layer | Alternate 128-token local and full causal attention | Save work in local layers; keep global access |
| Feed-forward path | One dense GELU FFN per block | Routed SwiGLU experts per block | Increase stored capacity with limited per-token work |
| Attention sink | Ordinary softmax over allowed tokens | Learned sink score per query head | Allow attention to allocate weight away from real tokens |

## Sources

- **GPT-2 vs GPT-OSS:** [Sebastian Raschka: From GPT-2 to gpt-oss](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the)
- **MoE intuition:** [Hugging Face: Mixture of Experts Explained](https://huggingface.co/blog/moe); [Mixture of Experts in Transformers](https://huggingface.co/blog/moe-transformers)
- **Model details:** [GPT-2 report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [GPT-2 code](https://github.com/openai/gpt-2/blob/master/src/model.py), [gpt-oss model card](https://arxiv.org/abs/2508.10925), [gpt-oss code](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py)
- **Core methods:** [RoPE](https://arxiv.org/abs/2104.09864), [RMSNorm](https://arxiv.org/abs/1910.07467), [GQA](https://arxiv.org/abs/2305.13245), [GLU variants](https://arxiv.org/abs/2002.05202), [Switch Transformers](https://arxiv.org/abs/2101.03961)
