# Reinforcement Learning

## Reinforcement Learning Problem Statement

Unlike supervised learning, there is **no dataset of correct answers**. An agent interacts with an environment and learns from the consequences of its actions.

At each timestep:

1. Observe state $s_t$
2. Choose action $a_t$
3. Receive reward $r_t$
4. Environment transitions to $s_{t+1}$

The objective is **not** to maximize the immediate reward, but the **expected cumulative discounted reward**.

The **return** is defined as

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1},
$$

where

- $0 \le \gamma < 1$ is the discount factor.

The optimization objective is

$$
\pi^*=\arg\max_\pi \mathbb E_\pi[G_0].
$$


### Markov Decision Process (MDP)

Classical reinforcement learning assumes interaction with a **Markov Decision Process (MDP)**

$$
(\mathcal S,\mathcal A,P,R,\gamma).
$$

where

- $\mathcal S$: state space
- $\mathcal A$: action space
- $P(s'|s,a)$: transition probability
- $R(s,a)$ or $R(s,a,s')$: reward function
- $\gamma$: discount factor

#### Markov Property

The environment satisfies

$$
P(s_{t+1}|s_t,a_t,\text{history})
=
P(s_{t+1}|s_t,a_t).
$$

The current state contains all information necessary for predicting the future.


### Policy

A **policy** specifies how the agent chooses actions.

#### Deterministic policy

$$
a=\pi(s)
$$

#### Stochastic policy

$$
\pi(a|s)=P(a|s).
$$

Most modern RL algorithms optimize stochastic policies because they naturally support exploration.


### Value Functions

#### State-value function

The expected future discounted reward starting from state $s$:

$$
V^\pi(s)
=
\mathbb E_\pi[G_t|s_t=s].
$$

Interpretation:

> "How good is this state if I follow policy $\pi$?"


#### Action-value function

The expected future reward after taking action $a$ in state $s$:

$$
Q^\pi(s,a)
=
\mathbb E_\pi[G_t|s_t=s,a_t=a].
$$

Interpretation:

> "How good is taking this action?"

Relationship:

- $V(s)$: evaluates states
- $Q(s,a)$: evaluates state-action pairs


## Bellman Expectation Equation

The Bellman equation follows from the recursive definition of return

$$
G_t=r_{t+1}+\gamma G_{t+1}.
$$

Substituting into the value function gives

$$
V^\pi(s)
=
\mathbb E[r+\gamma V^\pi(s')].
$$

Expanding the expectation,

$$
V^\pi(s)
=
\sum_a
\pi(a|s)
\sum_{s'}
P(s'|s,a)
\left[
R(s,a,s')
+
\gamma V^\pi(s')
\right].
$$

### Interpretation

The value of a state equals

- immediate reward,
- plus discounted value of the next state,

averaged over

- actions selected by the policy,
- stochastic environment transitions.


### Bellman Equation for Q

Similarly,

$$
Q^\pi(s,a)
=
\sum_{s'}
P(s'|s,a)
\left[
R(s,a,s')
+
\gamma
\sum_{a'}
\pi(a'|s')
Q^\pi(s',a')
\right].
$$

Interpretation:

The value of taking action $a$ equals

- immediate reward,
- plus discounted expected future value after following the policy.

---

### Optimal Value Functions

Instead of evaluating a fixed policy, suppose the agent always behaves optimally.

Define

$$
V^*(s)=\max_\pi V^\pi(s).
$$

The Bellman optimality equation becomes

$$
V^*(s)
=
\max_a
\sum_{s'}
P(s'|s,a)
\left[
R(s,a,s')
+
\gamma V^*(s')
\right].
$$

Unlike the expectation equation, there is no averaging over actions—the best action is always selected.

Similarly,

$$
Q^*(s,a)
=
\sum_{s'}
P(s'|s,a)
\left[
R(s,a,s')
+
\gamma
\max_{a'}
Q^*(s',a')
\right].
$$

This equation is the foundation of

- Value Iteration
- Q-learning
- Deep Q Networks (DQN)


### Bellman Expectation vs. Bellman Optimality

| Bellman Expectation | Bellman Optimality |
|---------------------|--------------------|
| Evaluates a fixed policy | Finds the optimal policy |
| Uses $\pi(a\|s)$ | Uses $\max_a$ |
| Policy evaluation | Control |
| Used in Policy Iteration | Used in Value Iteration, Q-learning, DQN |


## Why Bellman Equations Matter

Without Bellman equations, computing

$$
G=\sum_{t=0}^{\infty}\gamma^t r_t
$$

would require reasoning over complete future trajectories.

Bellman's key insight is

$$
\text{Value now}
=
\text{Reward now}
+
\text{Value later}.
$$

This recursive decomposition enables

- Dynamic Programming
- Temporal-Difference Learning
- Bootstrapping methods
- Most modern RL algorithms


## Policy Improvement Theorem

The **Policy Improvement Theorem** states that if we improve a policy by acting greedily with respect to its action-value function, the new policy is **at least as good as** the original one.

Suppose we have a policy $\pi$ and its action-value function $Q^\pi(s,a)$. Define a new policy

$$
\pi'(s)=\arg\max_a Q^\pi(s,a).
$$

Then, for every state $s$,

$$
V^{\pi'}(s) \ge V^\pi(s).
$$

In other words, the new policy performs no worse than the original, and is strictly better if there exists at least one state where the greedy action differs from the old policy.

### Intuition

The current policy $\pi$ tells us how good each action is via $Q^\pi(s,a)$. If another action has a higher expected return, choosing it cannot decrease performance. Repeating this process gradually improves the policy until it becomes optimal.

### Policy Iteration

The theorem motivates the classic **Policy Iteration** algorithm:

1. **Policy Evaluation:** Estimate $V^\pi$ or $Q^\pi$ for the current policy.
2. **Policy Improvement:** Update the policy greedily:

$$
\pi_{\text{new}}(s)=\arg\max_a Q^\pi(s,a).
$$

3. Repeat until the policy no longer changes.

At convergence,

$$
\pi_{\text{new}} = \pi,
$$

meaning the policy is greedy with respect to its own value function and is therefore optimal.

### In Deep Reinforcement Learning

In practice, $Q^\pi(s,a)$ is rarely known exactly. Instead, it is **estimated** using a function approximator (typically a neural network). The policy is then updated to choose actions with the highest estimated value:

$$
\pi(s)=\arg\max_a \hat{Q}(s,a).
$$

Because $\hat{Q}$ is only an approximation, policy improvement is no longer guaranteed at every iteration, but this principle remains the foundation of value-based methods such as Q-learning and Deep Q Networks (DQN).


## Monte Carlo Methods

Monte Carlo (MC) methods estimate value functions by **averaging complete returns from sampled episodes**, without requiring knowledge of the environment dynamics.

Unlike Dynamic Programming, which relies on the transition model $P(s'|s,a)$, Monte Carlo methods learn directly from experience.


### Monte Carlo Policy Evaluation

Given a policy $\pi$, the state-value function is estimated by averaging observed returns:

$$
V^\pi(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i(s),
$$

where

- $N(s)$ is the number of visits to state $s$,
- $G_i(s)$ is the total discounted return observed after the $i$-th visit.

Similarly,

$$
Q^\pi(s,a)
$$

is estimated by averaging returns after taking action $a$ in state $s$.


### First-Visit vs Every-Visit MC

Two common variants are:

- **First-Visit MC:** Update a state (or state-action pair) only the first time it appears in an episode.
- **Every-Visit MC:** Update every occurrence within the episode.

Both converge to the true value under sufficient exploration.


### Monte Carlo Control

To learn an optimal policy:

1. Generate episodes using the current policy.
2. Estimate $Q(s,a)$ from observed returns.
3. Improve the policy greedily:

$$
\pi(s)=\arg\max_a Q(s,a).
$$

Repeating evaluation and improvement implements **Generalized Policy Iteration (GPI)**.

To ensure all actions continue to be explored, the policy is usually made **$\varepsilon$-greedy**:

- with probability $1-\varepsilon$: choose the greedy action,
- with probability $\varepsilon$: choose a random action.

### Advantages

- Does not require a model of the environment.
- Produces unbiased estimates of returns.
- Simple and easy to implement.
- Naturally handles stochastic environments.

### Limitations

- Requires complete episodes before updating.
- High variance due to random returns.
- Cannot naturally learn from continuing (non-terminating) tasks.
- Typically less sample-efficient than Temporal Difference (TD) methods.

### Monte Carlo vs Dynamic Programming

| Dynamic Programming | Monte Carlo |
|---------------------|-------------|
| Requires known transition model | Learns directly from experience |
| Uses Bellman equations | Uses sampled returns |
| Bootstraps from current value estimates | Waits until episode ends |
| Lower variance | Higher variance |
| Limited to known MDPs | Works without a model |


## On-Policy vs Off-Policy Monte Carlo

Monte Carlo methods can learn either **from the same policy used to generate data (on-policy)** or **from data generated by a different policy (off-policy)**.

### On-Policy Monte Carlo

The **behavior policy** (used to collect experience) and the **target policy** (being evaluated or improved) are the same.

$$
\pi_{\text{behavior}} = \pi_{\text{target}}.
$$

A common approach is **$\varepsilon$-greedy Monte Carlo control**:

1. Generate episodes using the current $\varepsilon$-greedy policy.
2. Estimate $Q(s,a)$ by averaging returns.
3. Update the policy to be $\varepsilon$-greedy with respect to the new $Q$.
4. Repeat.

Exploration is maintained because every action has a non-zero probability of being selected.


### Off-Policy Monte Carlo

The data is collected using a **behavior policy** $b$, while learning is performed for a different **target policy** $\pi$:

$$
b \neq \pi.
$$

This allows learning an optimal (often deterministic) policy while collecting data with a more exploratory policy.

Since the observed trajectories were generated by $b$, their returns must be corrected using **importance sampling**.


### Importance Sampling

Each episode is weighted by the likelihood ratio

$$
\rho
=
\prod_{t=0}^{T-1}
\frac{\pi(a_t|s_t)}
     {b(a_t|s_t)}.
$$

The return is then weighted:

$$
V^\pi(s)
\approx
\frac{1}{N}
\sum_i
\rho_i G_i.
$$

Intuition:

- If a trajectory is likely under the target policy, it receives a larger weight.
- If it is unlikely, its contribution is reduced.


### Ordinary vs Weighted Importance Sampling

#### Ordinary Importance Sampling

$$
V(s)
=
\frac{1}{N}
\sum_i
\rho_i G_i.
$$

- Unbiased estimate.
- Can have very high variance.

#### Weighted Importance Sampling

$$
V(s)
=
\frac{\sum_i \rho_i G_i}
{\sum_i \rho_i}.
$$

- Slightly biased for finite samples.
- Much lower variance.
- Preferred in practice.


### On-Policy vs Off-Policy

| On-Policy | Off-Policy |
|-----------|------------|
| Learns from the policy generating data | Learns from a different target policy |
| No importance sampling | Requires importance sampling |
| Simpler and lower variance | More flexible but higher variance |
| Typically uses $\varepsilon$-greedy exploration | Can learn a deterministic greedy policy while exploring with another policy |


## Temporal Difference (TD) Learning

Temporal Difference (TD) learning combines ideas from **Monte Carlo** and **Dynamic Programming**. Like Monte Carlo, it learns directly from experience without requiring a model of the environment. Like Dynamic Programming, it **bootstraps** by updating value estimates using other estimated values.


### TD(0) Policy Evaluation

Instead of waiting until the end of an episode, TD updates the value estimate **after every transition**.

Given a transition

$$
(s_t, a_t, r_{t+1}, s_{t+1}),
$$

the update rule is

$$
V(s_t)
\leftarrow
V(s_t)
+
\alpha
\left[
r_{t+1}
+
\gamma V(s_{t+1})
-
V(s_t)
\right],
$$

where

- $\alpha$ is the learning rate,
- $\gamma$ is the discount factor.

The term inside the brackets is called the **TD error**:

$$
\delta_t
=
r_{t+1}
+
\gamma V(s_{t+1})
-
V(s_t).
$$


### Intuition

Instead of using the true return

$$
G_t=r_{t+1}+\gamma r_{t+2}+\cdots,
$$

TD approximates it with the **one-step target**

$$
r_{t+1}+\gamma V(s_{t+1}).
$$

Thus, value estimates are updated immediately after each interaction, without waiting for the episode to finish.


### Bootstrapping

The defining characteristic of TD learning is **bootstrapping**:

> The algorithm updates an estimate using another estimate.

Unlike Monte Carlo methods, TD does **not** require complete returns.

### Advantages

- Learns online after every step.
- Does not require a model of the environment.
- Works for both episodic and continuing tasks.
- Typically more sample-efficient than Monte Carlo.

### Limitations

- Updates are biased because targets depend on current value estimates.
- Can be less stable when combined with function approximation.

### Monte Carlo vs TD

| Monte Carlo | Temporal Difference |
|--------------|---------------------|
| Waits until episode ends | Updates after every step |
| Uses actual return $G_t$ | Uses bootstrapped target $r+\gamma V(s')$ |
| No bootstrapping | Bootstrapping |
| Unbiased, high variance | Biased, lower variance |
| Episodic tasks only | Episodic and continuing tasks |


## TD Learning for Action-Value Functions

While TD prediction estimates the **state-value function** $V^\pi(s)$, most model-free control algorithms learn the **action-value function** $Q(s,a)$, since it directly enables policy improvement.

The TD update is analogous to that for $V(s)$, but applied to state-action pairs.

### TD Prediction for $Q(s,a)$

Given a transition

$$
(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1}),
$$

the update rule is

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
\left[
r_{t+1}
+
\gamma Q(s_{t+1},a_{t+1})
-
Q(s_t,a_t)
\right].
$$

The TD error is

$$
\delta_t
=
r_{t+1}
+
\gamma Q(s_{t+1},a_{t+1})
-
Q(s_t,a_t).
$$

Unlike Monte Carlo, the update is performed **after every transition**, without waiting for the episode to end.

### Policy Improvement

Once $Q(s,a)$ has been updated, the policy can be improved by acting greedily:

$$
\pi(s)
=
\arg\max_a Q(s,a).
$$

In practice, an **$\varepsilon$-greedy** policy is typically used to balance exploration and exploitation:

- with probability $1-\varepsilon$: choose the greedy action,
- with probability $\varepsilon$: choose a random action.

### The On-Policy vs Off-Policy Difference

The crucial distinction between TD control algorithms is **how the TD target is computed**.

#### On-policy (SARSA)

The target uses the **actual next action** taken by the current policy:

$$
r+\gamma Q(s',a').
$$

Since $a'$ is sampled from the current (e.g., $\varepsilon$-greedy) policy, the algorithm learns the value of **the policy being executed**.

Update:

$$
Q(s,a)
\leftarrow
Q(s,a)
+
\alpha
\left[
r+\gamma Q(s',a')-Q(s,a)
\right].
$$

#### Off-policy (Q-learning)

The target assumes the **best possible next action**, regardless of which action is actually taken:

$$
r+\gamma\max_{a'}Q(s',a').
$$

Update:

$$
Q(s,a)
\leftarrow
Q(s,a)
+
\alpha
\left[
r+\gamma\max_{a'}Q(s',a')-Q(s,a)
\right].
$$

The behavior policy may still be $\varepsilon$-greedy for exploration, but the learned policy is always greedy.


### SARSA vs Q-learning

| SARSA | Q-learning |
|--------|------------|
| On-policy | Off-policy |
| Uses actual next action $a'$ | Uses best next action $\arg\max_a Q(s',a)$ |
| Learns the value of the behavior policy | Learns the value of the optimal policy |
| More conservative during exploration | More optimistic and often learns faster |


## Policy Gradient Methods

Unlike value-based methods (e.g., SARSA or Q-learning), **policy gradient methods optimize the policy directly**, without first learning a value function to derive a policy.

Instead of learning

$$
Q(s,a),
$$

we parameterize the policy with parameters $\theta$ (typically a neural network):

$$
\pi_\theta(a|s).
$$

The objective is to find parameters that maximize the expected return.


### Objective Function

Define the expected discounted return under the policy:

$$
J(\theta)
=
\mathbb E_{\tau \sim \pi_\theta}
\left[
G_0
\right],
$$

where

- $\tau = (s_0,a_0,s_1,a_1,\ldots)$ is a trajectory generated by the policy,
- $G_0$ is the total return.

The optimization problem is simply

$$
\theta^*
=
\arg\max_\theta J(\theta).
$$


### Gradient Ascent

Since we want to **maximize** the objective, the parameters are updated using gradient ascent:

$$
\theta
\leftarrow
\theta
+
\alpha
\nabla_\theta J(\theta).
$$

The challenge is computing the gradient.


### Policy Gradient Theorem

A fundamental result in RL is the **Policy Gradient Theorem**:

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\pi_\theta}
\left[
\nabla_\theta
\log \pi_\theta(a|s)
\;
Q^{\pi_\theta}(s,a)
\right].
$$

This is remarkable because **it does not require differentiating through the environment dynamics**.

Instead, the gradient depends only on

- the policy,
- its log-probability,
- and how good the chosen action was.


### Intuition

Suppose an action led to a high return.

Then

$$
Q(s,a) > 0,
$$

so the update increases

$$
\log \pi_\theta(a|s),
$$

making that action **more likely** in the future.

Conversely, actions with poor returns receive negative updates, decreasing their probability.

In short:

- **Good actions become more probable.**
- **Bad actions become less probable.**

## REINFORCE Algorithm

The simplest policy gradient algorithm is **REINFORCE**.

After generating a complete episode, every action is updated according to the observed return:

$$
\theta
\leftarrow
\theta
+
\alpha
\nabla_\theta
\log \pi_\theta(a_t|s_t)
\, G_t.
$$

Here,

- $G_t$ is the actual return following action $a_t$,
- no value function is learned.

REINFORCE is therefore a **Monte Carlo policy gradient** method.


### Why It Works

The term

$$
\nabla_\theta \log \pi_\theta(a|s)
$$

points in the direction that **increases the probability of the chosen action**.

Multiplying by the return scales the update:

- large positive return → strongly increase the action probability,
- small or negative return → decrease its probability.

This idea is known as the **likelihood ratio trick** (or **score function estimator**).


### Advantages

- Directly optimizes the policy.
- Naturally handles stochastic policies.
- Works well in continuous action spaces.
- Avoids the $\max$ operator that makes value-based methods difficult to extend to continuous actions.


### Limitations

- High variance gradient estimates.
- Requires many samples.
- Typically less sample-efficient than value-based methods.
- Usually combined with a value function (Actor-Critic) to reduce variance.


### Value-Based vs Policy Gradient

| Value-Based Methods | Policy Gradient Methods |
|----------------------|-------------------------|
| Learn $Q(s,a)$ or $V(s)$ | Learn policy $\pi_\theta(a\|s)$ directly |
| Policy derived using $\arg\max_a Q(s,a)$ | Policy is the optimization target |
| Well suited for discrete actions | Naturally handles continuous actions |
| Examples: Q-learning, DQN | Examples: REINFORCE, PPO, TRPO |



## REINFORCE with Baseline

The original **REINFORCE** algorithm estimates the policy gradient using the observed return:

$$
\theta
\leftarrow
\theta
+
\alpha
\nabla_\theta \log \pi_\theta(a_t|s_t)
\, G_t.
$$

Although this estimator is **unbiased**, it suffers from **high variance**, making learning slow and unstable.


### Motivation for a Baseline

Suppose two actions both produce a return of 100.

- If 100 is **much better than expected**, we should strongly reinforce the action.
- If 100 is **exactly what we expected**, the update should be much smaller.

This suggests comparing the return to a **baseline** rather than using the raw return.


### Baseline Policy Gradient

A baseline $b(s)$ can be subtracted from the return without changing the expected gradient:

$$
\nabla_\theta J(\theta)
=
\mathbb E
\left[
\nabla_\theta
\log\pi_\theta(a|s)
\left(
G_t-b(s)
\right)
\right].
$$

A key theoretical result is that **any baseline independent of the chosen action does not introduce bias**. Its purpose is solely to **reduce variance**.


### Choosing the Baseline

The most common choice is the **state-value function**:

$$
b(s)=V^\pi(s).
$$

The update becomes

$$
\theta
\leftarrow
\theta
+
\alpha
\nabla_\theta
\log\pi_\theta(a_t|s_t)
\left(
G_t-V(s_t)
\right).
$$

Now the policy is updated based on how much better or worse the observed return was compared to the expected return.


### Advantage Function

The term

$$
A(s,a)
=
Q(s,a)-V(s)
$$

is called the **advantage function**.

Since REINFORCE uses the Monte Carlo return as an estimate of $Q$,

$$
Q(s_t,a_t)
\approx
G_t,
$$

the quantity

$$
G_t-V(s_t)
$$

is a Monte Carlo estimate of the advantage.

Thus, the update can be interpreted as

$$
\theta
\leftarrow
\theta
+
\alpha
\nabla_\theta
\log\pi_\theta(a_t|s_t)
\,A(s_t,a_t).
$$


### Learning the Baseline

Unlike the policy, the baseline is learned as a **regression problem**.

A value network $V(s;\mathbf w)$ is trained to predict the expected return:

$$
L(\mathbf w)
=
\left(
G_t-V(s_t;\mathbf w)
\right)^2.
$$

After each episode:

1. Compute Monte Carlo returns $G_t$.
2. Update the value network using MSE loss.
3. Update the policy using the advantage estimate $G_t-V(s_t)$.


### REINFORCE vs REINFORCE with Baseline

| REINFORCE | REINFORCE with Baseline |
|------------|-------------------------|
| Uses return $G_t$ | Uses advantage estimate $G_t - V(s_t)$ |
| No value function | Learns a value function $V(s)$ |
| High variance | Lower variance |
| Monte Carlo policy gradient | Monte Carlo Actor-Critic (critic trained from full returns) |


## Actor-Critic Methods

Actor-Critic methods combine the strengths of **policy-based** and **value-based** reinforcement learning.

Instead of learning only a policy (REINFORCE) or only a value function (Q-learning), they learn **both simultaneously**.

- **Actor:** learns the policy.
- **Critic:** evaluates the policy.


### Architecture

The two components are:

#### Actor

The actor parameterizes the policy

$$
\pi_\theta(a|s),
$$

and is responsible for selecting actions.

Its objective is to maximize the expected return.


#### Critic

The critic estimates the value of states or state-action pairs:

$$
V_w(s)
\quad\text{or}\quad
Q_w(s,a),
$$

where $w$ denotes the critic's parameters.

The critic does **not** choose actions—it only evaluates how good they are.


### Why Do We Need a Critic?

Vanilla REINFORCE uses the Monte Carlo return

$$
G_t
$$

to estimate

$$
Q^\pi(s,a).
$$

Although unbiased, this estimate has **high variance**.

Instead, the critic learns an approximation

$$
V_w(s)
\quad\text{or}\quad
Q_w(s,a),
$$

using **Temporal Difference (TD) learning**, providing lower-variance estimates.


### Critic Update

For a value critic, the TD target is

$$
y
=
r_t
+
\gamma
V_w(s_{t+1}),
$$

and the critic minimizes

$$
L(w)
=
\left(
y
-
V_w(s_t)
\right)^2.
$$

Thus, the critic is trained exactly like a regression model, where the target is the TD estimate.


### Actor Update

Instead of weighting the policy gradient by the full return, the actor uses the **advantage estimate**:

$$
A_t
=
r_t
+
\gamma
V_w(s_{t+1})
-
V_w(s_t).
$$

This is simply the **TD error**.

The policy update becomes

$$
\theta
\leftarrow
\theta
+
\alpha
\nabla_\theta
\log\pi_\theta(a_t|s_t)
A_t.
$$

Intuition:

- $A_t > 0$: the action performed **better than expected**, so increase its probability.
- $A_t < 0$: the action performed **worse than expected**, so decrease its probability.


### Training Loop

At every interaction:

1. Observe transition

$$
(s_t,a_t,r_t,s_{t+1}).
$$

2. Critic computes the TD error

$$
\delta_t
=
r_t
+
\gamma
V(s_{t+1})
-
V(s_t).
$$

3. Update the critic using TD learning.

4. Update the actor using

$$
\nabla_\theta
\log\pi(a_t|s_t)
\,\delta_t.
$$

Unlike REINFORCE, there is **no need to wait for the episode to finish**.


### Relationship to REINFORCE

REINFORCE with baseline updates the actor using

$$
G_t-V(s_t),
$$

where the baseline is learned from **Monte Carlo returns**.

Actor-Critic replaces the Monte Carlo return with the **TD target**:

$$
G_t
\quad\longrightarrow\quad
r_t+\gamma V(s_{t+1}).
$$

As a result:

- updates happen after every step,
- variance is much lower,
- learning is more sample-efficient.


### Advantages

- Lower variance than REINFORCE.
- Online updates after every transition.
- More sample-efficient.
- Naturally supports continuous action spaces.
- Foundation of most modern policy optimization algorithms.


### Limitations

- Introduces bias due to bootstrapping.
- Requires training two models simultaneously.
- Can become unstable if the critic learns poor value estimates.


### REINFORCE vs Actor-Critic

| REINFORCE | Actor-Critic |
|------------|--------------|
| Learns only the policy | Learns both policy and value function |
| Uses Monte Carlo return $G_t$ | Uses TD estimates |
| Updates after full episodes | Updates after every step |
| High variance | Lower variance |
| Unbiased | Slightly biased due to bootstrapping |
| Simpler | More sample-efficient |


### Relationship to Modern Algorithms

Many popular RL algorithms are Actor-Critic methods:

- **A2C (Advantage Actor-Critic):** synchronous actor-critic.
- **A3C (Asynchronous Advantage Actor-Critic):** multiple actors collect experience in parallel.
- **PPO:** Actor-Critic with a clipped policy objective.
- **TRPO:** Actor-Critic with a trust-region constraint.
- **DDPG:** Actor-Critic for continuous deterministic policies.
- **SAC (Soft Actor-Critic):** entropy-regularized Actor-Critic.
- **TD3:** improved deterministic Actor-Critic.


## Modern Policy Optimization Algorithms

The main limitation of vanilla policy gradient methods (e.g., REINFORCE) is that **large policy updates can drastically degrade performance**. Modern policy optimization algorithms aim to improve stability by constraining or regularizing policy updates.

### TRPO (Trust Region Policy Optimization)

**Key idea:** Prevent the policy from changing too much in a single update.

TRPO formulates policy optimization as a constrained optimization problem:

$$
\max_\theta J(\theta)
$$

subject to

$$
D_{\mathrm{KL}}(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta)
\leq \delta,
$$

where $D_{\mathrm{KL}}$ is the KL divergence between the old and new policies.

#### Advantages

- Monotonic policy improvement (approximately).
- Very stable training.

#### Limitations

- Computationally expensive.
- Requires second-order optimization (Fisher information matrix / conjugate gradient).
- Difficult to implement.


### PPO (Proximal Policy Optimization)

**Key idea:** Approximate TRPO with a much simpler first-order method.

Instead of enforcing a hard KL constraint, PPO clips the probability ratio

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\text{old}}}(a_t|s_t)}.
$$

The clipped objective is

$$
L^{\text{PPO}}
=
\mathbb E
\left[
\min
\left(
r_t A_t,\;
\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\right].
$$

#### Intuition

- If the new policy differs only slightly from the old one, perform a normal policy gradient update.
- If the update becomes too large, clip the objective to prevent overly aggressive changes.

#### Advantages

- Simple to implement.
- Stable training.
- Excellent empirical performance.
- One of the most widely used RL algorithms.


### GRPO (Group Relative Policy Optimization)

GRPO was introduced by **DeepSeek** as an efficient alternative to PPO for training large language models.

Instead of learning a separate value network (critic), GRPO computes advantages by comparing responses **within the same sampled group**.

For a prompt:

1. Sample multiple responses.
2. Score each response.
3. Normalize scores within the group.
4. Use the relative scores as advantages.

Thus,

$$
A_i
=
R_i
-
\text{group average}.
$$

#### Key idea

The group itself provides the baseline, eliminating the need for a learned critic.

#### Advantages

- No value network.
- Lower memory usage.
- Simpler optimization.
- Well suited for LLM post-training.


### Comparison

| Algorithm | Main Idea | Critic | Typical Use |
|-----------|-----------|--------|-------------|
| **TRPO** | Constrain policy updates using KL divergence | ✓ | Classical RL |
| **PPO** | Clip policy updates instead of constrained optimization | ✓ | Standard deep RL, RLHF |
| **GRPO** | Estimate advantages from groups of sampled responses | ✗ | LLM post-training |


## Reinforcement Learning Summary

| Concept | Goal | Learns | Update Method | Policy Type | Key Idea | Pros | Cons |
|----------|------|---------|---------------|-------------|----------|------|------|
| **Bellman Equations** | Define recursive value functions | $V(s)$, $Q(s,a)$ | Recursive equations | N/A | Value = immediate reward + discounted future value | Foundation of RL | Requires solving recursively |
| **Policy Iteration** | Find optimal policy | $V(s)$ (or $Q(s,a)$) + policy | Alternate policy evaluation and greedy improvement | Deterministic greedy | Repeatedly evaluate and improve policy | Guaranteed convergence (finite MDPs) | Requires known environment model |
| **Monte Carlo Estimation** | Evaluate a fixed policy | $V(s)$ or $Q(s,a)$ | Average complete returns | Fixed policy | Learn from complete episodes | Unbiased estimates, model-free | High variance, must wait until episode ends |
| **Monte Carlo Control** | Learn optimal policy | $Q(s,a)$ | MC estimation + greedy/$\varepsilon$-greedy improvement | Usually $\varepsilon$-greedy | Alternate evaluation and policy improvement | Model-free, simple | Sample inefficient, episodic only |
| **On-Policy MC** | Learn current policy | $Q(s,a)$ | Average sampled returns | Same behavior and target policy | Learn from actions actually taken | Simple, stable | Exploration affects learned policy |
| **Off-Policy MC** | Learn different policy | $Q(s,a)$ | Importance sampling | Different behavior and target policies | Learn one policy while executing another | Can reuse exploratory data | Importance sampling has high variance |
| **TD Policy Evaluation** | Evaluate a fixed policy | $V(s)$ | $TD(0)$: $r+\gamma V(s')$ | Fixed policy | Bootstrapping (estimate from estimate) | Online learning, lower variance | Biased target |
| **SARSA** | Learn optimal policy | $Q(s,a)$ | TD update using actual next action | **On-policy** | Learn value of behavior policy | Safe, conservative | Exploration affects learned values |
| **Q-Learning** | Learn optimal policy | $Q(s,a)$ | TD update using greedy next action | **Off-policy** | Learn value of optimal policy | Efficient, foundation of DQN | Can overestimate action values |
| **Policy Gradient** | Optimize policy directly | Policy $\pi_\theta(a\|s)$ | Gradient ascent on expected return | Stochastic | Increase probability of good actions | Continuous actions, direct optimization | High variance |
| **REINFORCE** | Learn policy | Policy only | Monte Carlo policy gradient | On-policy | Use complete return as gradient weight | Unbiased gradient | Very high variance, episodic |
| **REINFORCE + Baseline** | Reduce gradient variance | Policy + value baseline | MC return minus baseline | On-policy | Learn from advantage $G-V$ | Lower variance, more stable | Still Monte Carlo, episodic |
| **Actor-Critic** | Optimize policy with TD critic | Policy + critic $V$ or $Q$ | TD critic + policy gradient | Usually on-policy (can be off-policy) | Critic provides low-variance learning signal | Sample efficient, online updates | Biased, more complex |
| **TRPO** | Stable policy optimization | Actor + critic | Trust-region constrained optimization | On-policy | Limit KL divergence between policies | Very stable updates | Complex second-order optimization |
| **PPO** | Practical stable policy optimization | Actor + critic | Clipped policy objective | On-policy | Prevent excessively large policy updates | Simple, robust, state-of-the-art baseline | Slightly less theoretically grounded than TRPO |
| **GRPO** | Efficient LLM policy optimization | Policy only (group baseline) | Relative group advantages | On-policy | Compare sampled responses instead of learning a critic | No value network, memory efficient | Primarily designed for LLM post-training |


## High-Level Evolution of RL Algorithms

```text
Bellman Equations
        │
        ▼
Dynamic Programming
(Policy Iteration / Value Iteration)
        │
        ├────────────── Requires Environment Model
        │
        ▼
Model-Free Reinforcement Learning
        │
        ├────────────── Monte Carlo
        │                  │
        │                  ├── Policy Evaluation
        │                  ├── Monte Carlo Control
        │                  ├── On-Policy MC
        │                  └── Off-Policy MC
        │
        └────────────── Temporal Difference
                           │
                           ├── TD Policy Evaluation
                           ├── SARSA (On-Policy)
                           └── Q-Learning (Off-Policy)
                                   │
                                   ▼
                              Deep Q-Networks (DQN)

──────────────────────────────────────────────────────

Policy-Based Reinforcement Learning
        │
        ▼
Policy Gradient Theorem
        │
        ▼
REINFORCE (Monte Carlo)
        │
        ▼
REINFORCE + Baseline
        │
        ▼
Actor-Critic (TD Critic)
        │
        ├── TRPO
        └── PPO

──────────────────────────────────────────────────────

LLM Alignment

PPO (RLHF)
      │
      └── GRPO (critic-free relative advantage)
```
