# RLHF

## Motivation

A pretrained LLM is trained to predict the next token:

$$
\max_\theta \log P_\theta(x).
$$

This objective does **not** optimize for qualities such as

- helpfulness,
- harmlessness,
- honesty,
- reasoning quality,
- code quality.

Moreover, for many prompts there is **no single correct answer**.

Instead of asking humans to write the perfect response, it is much easier to ask them

> "Which response is better?"

This leads to **preference learning**.

### Supervised Fine-Tuning (SFT)

Starting from a pretrained model

$$
\pi_{\text{pretrain}},
$$

we fine-tune on demonstration data.

This produces

$$
\pi_{\text{SFT}},
$$

which serves as the **initial policy**.

Without SFT, reinforcement learning would be extremely unstable because the model initially generates mostly poor responses.


### Collect Preference Data

For every prompt

$$
x,
$$

sample several responses

$$
y_1,y_2,\ldots
$$

using the SFT model.

Humans rank them.

The dataset consists of

$$
(x,y^+,y^-),
$$

where

- $y^+$ is preferred,
- $y^-$ is rejected.

Notice:

**there are no numerical rewards.**

We only know

$$
y^+ \succ y^-.
$$


### Reward Model

The reward model

$$
R_\phi(x,y)
$$

outputs a scalar

```
Prompt + Response

↓

Reward Model

↓

Reward = 4.2
```

#### How is the reward model trained?

This is essentially a **binary classification problem**.

The model should predict that

$$
y^+
$$

is preferred over

$$
y^-.
$$

Instead of predicting a class directly, the reward model predicts two scalar scores

$$
R(x,y^+)
$$

and

$$
R(x,y^-).
$$

We then model

$$
P(y^+ \succ y^-)
=
\sigma
\left(
R(x,y^+)-R(x,y^-)
\right),
$$

where $\sigma$ is the sigmoid.

This is the **Bradley-Terry model**.

The loss becomes

$$
L
=
-
\log
\sigma
(
R^+-R^-
).
$$


## PPO Optimization

Now the reward model is frozen.

The policy generates a response

```
Prompt

↓

LLM

↓

Response

↓

Reward Model

↓

Reward
```

The reward is treated exactly like the reward in reinforcement learning.

The objective becomes

$$
\max_\theta
\mathbb E
[
R(x,y)
].
$$


### Why Reinforcement Learning?

This often confuses people because the "environment" is unusual.

The mapping is

| Classical RL | RLHF |
|--------------|------|
| State | Prompt + generated prefix |
| Action | Next generated token |
| Policy | LLM |
| Episode | Complete generated response |
| Reward | Reward model score |

Generating text is a **sequential decision-making process**.

Each generated token changes the future context and therefore affects future token choices and the final reward.

The reward is only observed **after the entire response has been generated**, making this an episodic RL problem.


### Connection to Policy Gradients

Suppose the response is

$$
y=(a_1,a_2,\ldots,a_T).
$$

The probability of the response is

$$
\pi(y|x)
=
\prod_t
\pi(a_t|x,a_{<t}).
$$

The policy gradient objective is

$$
\nabla J
=
\mathbb E
\left[
\nabla
\log\pi(y|x)
R(y)
\right].
$$

Since

$$
\log\pi(y|x)
=
\sum_t
\log\pi(a_t|s_t),
$$

the update becomes

$$
\nabla J
=
\mathbb E
\left[
R(y)
\sum_t
\nabla
\log\pi(a_t|s_t)
\right].
$$

This is exactly **REINFORCE**:

- the generated response is an episode,
- every generated token is an action,
- the reward model provides the terminal reward.

Thus RLHF is simply **policy gradient on language generation**.


### Why PPO?

Vanilla REINFORCE has extremely high variance.

Actor-Critic reduces variance but still allows the policy to change drastically.

Suppose the reward model mistakenly gives high rewards to

```
"Thank you!!!!!!!!!"
```

Without constraints, the policy may quickly collapse into producing pathological responses.

PPO stabilizes learning by clipping policy updates.

Additionally, RLHF includes a KL penalty

$$
R'
=
R
-
\beta
D_{KL}
(
\pi_\theta
||
\pi_{\text{SFT}}
),
$$

preventing the model from drifting too far from the supervised model.

Therefore PPO provides

- lower variance,
- stable updates,
- controlled policy changes.


#### Which Parameters Are Updated?

During PPO training:

- the reward model is **frozen**,
- the reference (SFT) model is **frozen**,
- only the policy model is optimized.

The policy may be trained

- by updating **all parameters** (common in frontier model training), or
- using **PEFT methods** such as **LoRA** or **QLoRA** (common in research and open-source fine-tuning to reduce memory and compute).

The RL algorithm itself is independent of this choice—it simply computes gradients for the trainable parameters.


### Weaknesses of RLHF

RLHF requires

- collecting preference data,
- training a reward model,
- PPO optimization,
- actor and critic networks.

It is computationally expensive.

Moreover, the policy optimizes the **reward model**, not humans directly.

If the reward model has flaws, the policy may exploit them (**reward hacking**).


## DPO (Direct Preference Optimization)

The DPO paper starts from the same RLHF objective:

$$
\max_\pi
\mathbb E[R]
-
\beta
D_{KL}
(
\pi
||
\pi_{\text{ref}}
).
$$

The key insight is that this optimization problem has a closed-form optimum.

### Deriving DPO

The optimal policy satisfies

$$
\pi^*(y|x)
\propto
\pi_{\text{ref}}(y|x)
\exp
\left(
\frac{R(x,y)}{\beta}
\right).
$$

Taking logs,

$$
R(x,y)
=
\beta
\left(
\log\pi^*(y|x)
-
\log\pi_{\text{ref}}(y|x)
\right)
+
C,
$$

where $C$ is a normalization constant independent of $y$.

Now substitute this into the Bradley-Terry preference probability:

$$
P(y^+ \succ y^-)
=
\sigma
(R^+-R^-).
$$

The constant cancels:

$$
R^+-R^-
=
\beta
\left[
\log
\frac{\pi(y^+)}
{\pi(y^-)}
-
\log
\frac{\pi_{\text{ref}}(y^+)}
{\pi_{\text{ref}}(y^-)}
\right].
$$

Substituting into the Bradley-Terry loss gives

$$
L_{\text{DPO}}
=
-
\log
\sigma
\left(
\beta
\log
\frac{\pi(y^+)}
{\pi(y^-)}
-
\beta
\log
\frac{\pi_{\text{ref}}(y^+)}
{\pi_{\text{ref}}(y^-)}
\right).
$$

This is exactly the DPO objective.


### Intuition Behind DPO

Suppose

```
Current model

Preferred : 0.30

Rejected : 0.70
```

DPO directly increases

```
Preferred ↑

Rejected ↓
```

relative to the reference model.

Unlike RLHF, there is

- no reward model,
- no critic,
- no PPO.

Training becomes ordinary supervised optimization over preference pairs.


## RLHF vs DPO

| RLHF | DPO |
|------|-----|
| Reward model | No reward model |
| PPO optimization | Supervised optimization |
| Actor + Critic | Policy only |
| Online RL | Offline learning |
| More expensive | Simpler |
| Can optimize arbitrary rewards | Limited to preference datasets |
