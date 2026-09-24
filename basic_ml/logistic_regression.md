# Logistic Regression
1. From log odds to probability
- suppose we are predicting log of odds $\langle w, x \rangle - \log \frac{p}{1-p}$
- then $p$ is
$$
p = \frac{1}{1 + e^{-\langle w, x_ \rangle}} = \sigma(\langle w, x \rangle )
$$

2. Loss Function
- need to decide what to optimize
- aim at matching distribution of predicted probabilites $p$ with true labels distribution $y$
- optimize KL divergence
$$
KL(y \| p) = \sum y \log \frac{y}{p} = \sum y log y - \sum y log p = -H(y) + H(y, p)
$$
- H(y) is fixed
- we can optimize Cross Entropy $CE(y, p)$ and use it as a loss function (minimize it)

4. Maximum Likelihood
- Binomial distribution $p(y|x,w)=\sigma(\langle w, x \rangle)^y (1-\sigma(\langle w, x \rangle))^{1-y} = p^y (1-p)^{1-y}$
$$
ML=\prod p(y_i|x_i,w) = \prod p^{y_i} (1-p)^{1-y_i}\\

\log ML = \sum y_i \log p + (1-y_i) \log (1-p) = -CE(y, p)
$$
- came to the same conclusion - we need to minimize cross entropy in order to maximize likelihood

5. Optimization
$$
\nabla_w L(y, X, w) = - \sum y_i x_i - x_i \sigma (\langle w, x_i \rangle)
$$
- L is positive semidefinite

6. Generalized Linear Models
- link funcion $g(\mu) = \log (\mu / (1-\mu))$
- $\langle w, x \rangle = g(\mu)$
- $\mu = \mathbb{E}(y|x) = p = g^{-1}(\langle w, x \rangle) = \sigma (\langle w, x_i \rangle)$

7. Missclassification Loss, Perceptron
- predicted class $\mathrm{sign}\langle w, x \rangle$
- missclassification
$$
\sum [y_i \neq \mathrm{sign}\langle w, x_i \rangle] = \sum [y_i \mathrm{sign}\langle w, x_i \rangle < 0] = \sum [M_i<0]
$$
- $M = y \mathrm{sign}\langle w, x\rangle$ is so-called margin
- $F(M) = \max(0, -M)$ - perceptron Loss - produces non-unique solutions
- $F(M) = \max(0, 1-M)$ - Hinge (SVM) Loss - produces unique solutions

8. SVM
$$
\sum \max(0, 1-M_i) + \lambda \| w \|_2^2 \rightarrow \min
$$
- $\max_{\|w\|} \min d(x_i, A)$ - quadratic programming, $N$ variables
- $\min \| \bar{a} - \bar{b} \|$, $\bar{a}$, $\bar{b}$ are convex hulls - quadratic programmind, $d$ variables
- Lyapunov function with Lagrange multipliers
- Karush–Kuhn–Tucker conditions
$$
\begin{cases}
\alpha_i >=0 \\
y_i f(x_i) - 1 >= 0 \\
\alpha_i (y_i f(x_i) - 1) = 0
\end{cases}
$$
- Kernel Trick
    - everything depends only on dot product
    - expand features to higher dimensions
    - dot product in higher dimension is kernel function $K(x, y)$
