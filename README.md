# Yandex ML Interview Preparation Notes

These are the notes I used to prepare for interviews for a **Computer Vision / Deep Learning Engineer** role at Yandex. I received **grade 17 (Senior)** after the technical interviews. I am publishing the notes both as a record of that preparation and as a reference I can return to before future interviews.

This is a personal interview prep guide, not a complete ML textbook. Most of the notes are prompts for explaining a topic aloud: what to cover, which details are easy to forget, and how to keep the explanation coherent.

## Interview rounds and notes

| Round | What to expect | Notes in this repository |
| --- | --- | --- |
| **1. Machine Learning & Programming (MLP)** | A small programming task, plus a discussion of a familiar ML algorithm and the full model development process: data, training, tuning, and evaluation. | [Classical ML notes](#round-1-classical-ml) |
| **2. Algorithms & Data Structures** | Two problems around LeetCode medium difficulty. | Not covered here. |
| **3. Machine Learning Special** | A deeper discussion of the candidate's specialty; for me, computer vision and deep learning. | [Neural networks](nn.md) and [computer vision](cv.md) |
| **4. Final interview** | A discussion with a specific team, potentially focused on its domain. I did not know which topics to expect, so I prepared broadly. | [RL and alignment](#rl-and-alignment) and [LLM inference optimization](#llm-inference-optimization) |

### Round 1: Classical ML

I had interviewed at Yandex before, so I expected the ML discussion to use a favorite algorithm as a starting point for questions about the entire training pipeline. I wrote the [`basic_ml/`](basic_ml/) notes myself as short speaking outlines rather than detailed tutorials:

- [Linear regression](basic_ml/linear_regression.md)
- [Logistic regression](basic_ml/logistic_regression.md)
- [Decision trees](basic_ml/decision_trees.md)
- [Random forests](basic_ml/random_forest.md)
- [Gradient boosting](basic_ml/gradient_boosting.md)
- [Evaluation metrics](basic_ml/metrics.md)

Neural networks have their own section because they were less likely to come up in my first round. A/B testing is another possible topic, but it is not covered in these notes. The programming part of MLP is also outside the scope of this repository.

### Round 3: Neural networks and computer vision

- [**Neural networks**](nn.md) covers optimizers, regularization, and training issues such as vanishing or exploding gradients. I wrote it as a structured reminder of details I wanted to recall during an interview.
- [**Computer vision**](cv.md) surveys CV tasks, the development of model architectures from early CNNs to vision transformers, and serving and optimization topics. It includes illustrative code and references within the notes.

### Round 4: Broad preparation

I did not know what the final team's interview would cover, so I prepared across several areas. I was particularly interested in the YandexGPT alignment team, but also studied LLM inference optimization.

#### RL and alignment

- [**Reinforcement learning**](rl.md) starts with MDPs and Bellman equations, then builds through Monte Carlo and temporal difference methods, policy gradients, actor–critic methods, and modern policy optimization.
- [**RLHF**](rlhf.md) follows the alignment pipeline from supervised fine-tuning and preference data to reward models, PPO, and DPO.
I prepared the RL and RLHF notes with AI assistance and reviewed and organized them for my own study. Their progression from simpler ideas to more advanced algorithms was inspired by S. N. Nikolenko's lectures.

#### Modern LLM architecture
- [**GPT-2 → gpt-oss**](modern_llm.md) compares the decoder blocks, position encoding, normalization, attention, and mixture of experts.

#### LLM inference optimization

These notes focus on how a transformer generates tokens and how inference systems reduce latency or serve more requests:

- [**Transformer generation and KV caching**](inference_optimization/kv_cache.ipynb) explains prefill, decode, full-prefix recomputation, KV cache strategies, and measured generation time.
- [**PagedAttention**](inference_optimization/paged_attention.md) explains block-based KV storage, logical-to-physical mapping, prefix sharing, and memory tradeoffs.
- [**Attention backends**](inference_optimization/attention_backend.ipynb) compares the optimization ideas behind attention implementations and includes a timing notebook.
- [**Continuous batching**](inference_optimization/continious_batching.md) explains chunked prefill, ragged batching, and dynamic scheduling with token-budget and attention-mask diagrams.
- [**Tensor parallelism**](inference_optimization/tensor_parallelism.md) explains column/row splits, attention heads, and inference communication tradeoffs.
- [**Quantization**](inference_optimization/quantization.md) covers scales and zero points, granularity, packing, calibration, and inference memory/speed tradeoffs.
- [**Assisted generation**](inference_optimization/assisted_generation.md) covers draft-and-verify decoding, acceptance rules, speed tradeoffs, and drafting methods such as Medusa, MTP, and EAGLE.
- [**MoE inference optimization**](inference_optimization/moe_inference.md) explains expert kernels, expert parallelism, token exchange, load balancing, and prefill/decode tradeoffs.

## How to use these notes

Pick a topic and try to explain it out loud before opening its file. Use the notes to check the structure of your answer and catch missing points. For the later rounds, read [neural networks](nn.md) before [computer vision](cv.md), and [reinforcement learning](rl.md) before [RLHF](rlhf.md); each pair builds on the earlier material.

For inference optimization, start with [transformer generation and KV caching](inference_optimization/kv_cache.ipynb). Then use the other guides to study attention computation, request scheduling, and assisted decoding.
