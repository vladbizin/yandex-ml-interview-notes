# Random Forest

1. Bias-Variance Decomposition
- true value $y=f(x) + \varepsilon = y(x, \varepsilon)$, $\varepsilon \sim \mathcal{N}(0, \sigma^2)$
- predicted value $a(x) = a(x, X)$ is dependent on training dataset $X$
- prediction is dependent on
    * random noise $\varepsilon$
    * dataset $X$
    * sample object $x$
- average prediction error
$$
\mathbb{E}_x\mathbb{E}_{X, \varepsilon} [(y(x, \varepsilon)-a(x, X))^2]
$$

- for a single object
$$
\mathbb{E}_{X,\varepsilon}[(f(x) + \varepsilon - a(x, X))^2] = \mathbb{E}_X \mathbb{E}\varepsilon[(f(x) + \varepsilon - a(x, X))^2] = \\

= \mathbb{E}_X \mathbb{E}\varepsilon [(f(x)-a(x, X))^2 + 2 \varepsilon(f(x)-a(x, X)) + \varepsilon^2] = \\

= \mathbb{E}_X [(f(x)-a(x, X))^2] + \underbrace{\mathbb{E}\varepsilon[\varepsilon]}_{=0} \mathbb{E}_X [f(x)-a(x, X)] + \mathbb{E}\varepsilon[\varepsilon^2] = \\

= \mathbb{E}_X [(f(x)-a(x, X))^2] + \sigma^2
$$

- simplify further
$$
\mathbb{E}_X [(f(x)-a(x, X))^2] = \mathbb{E}_X [(f(x) - \mathbb{E}_X [a(x, X)] + \mathbb{E}_X [a(x, X)] - a(x, X))^2] = \\

\mathbb{E}_X [(f(x) - \mathbb{E}_X [a(x, X)])^2] + \underbrace{\mathbb{E}_X [(a(x, X) - \mathbb{E}_X [a(x, X)])^2]}_{\mathbb{V}_X[a(x, X)]} + 2 \mathbb{E}_X [(f(x) - \mathbb{E}_X [a(x, X)])(a(x, X) - \mathbb{E}_X [a(x, X)])] = \\

= (\underbrace{f(x) - \mathbb{E}_X [a(x, X)]}_{\text{bias}_X a(x, X)})^2 + \mathbb{V}_X[a(x, X)] + 2 (f(x) - \mathbb{E}_X [a(x, X)]) \underbrace{(\mathbb{E}_X [a(x, X)] - \mathbb{E}_X [a(x, X)])}_{=0} \\

= \text{bias}_X^2 a(x, X) + \mathbb{V}_X[a(x, X)]
$$

- bias $\text{bias}_X a(x, X) = f(x) - \mathbb{E}_X [a(x, X)]$ - systematic error from model underfitting
- variance $\mathbb{V}_X[a(x, X)] = \mathbb{E}_X [(a(x, X) - \mathbb{E}_X [a(x, X)])^2]$ - how predictions for object vary across different training sets
- irreducible error $\sigma^2 = \mathbb{E}_{\varepsilon}[\varepsilon^2]$

2. Bagging (Bootstrap Aggregation)
- sample N objects with replacement from dataset
- probability of not picking a sample
$$
(1 - \frac{1}{N})^N \rightarrow \frac{1}{e} \approx 37%
$$
- each bootstrap contains $\approx 63%$ of objects from initial dataset

3. Bagging Bias-Variance
- train basic alogithm $b$ on bootstrapped dataset $k$ times and average
$$
a(x) = \frac{1}{k} \sum b(x, X_i)
$$
- bias is preserved
$$
\text{bias}_X a(x, X) = f(x) - \mathbb{E}_X [a(x, X)] = f(x) - \mathbb{E}_X \Big[ \frac{1}{k} \sum b(x, X_i) \Big] = \\

= f(x) - \frac{1}{k} \mathbb{E}_X \Big[\sum b(x, X_i) \Big] = f(x) - \frac{1}{k} \sum \mathbb{E}_X [b(x, X_i)] \approx \\

\approx f(x) - \frac{1}{k} \sum \mathbb{E}_X [b(x, X)] = f(x) - \mathbb{E}_X [b(x, X)] = \text{bias}_X b(x, X)
$$
- variance linearly decreases
$$
\mathbb{V}_X[a(x, X)] = \mathbb{E}_X [(a(x, X) - \mathbb{E}_X [a(x, X)])^2] =
\mathbb{E}_X \Big[ \frac{1}{k} \sum b(x, X_i) - \mathbb{E}_X \Big[ \frac{1}{k} \sum b(x, X_i) \Big] \Big]^2 = \\

= \frac{1}{k^2} \mathbb{E}_X \Big[\sum b(x, X_i) - \mathbb{E}_X \Big[\sum b(x, X_i) \Big] \Big]^2 =
\frac{1}{k^2} \mathbb{E}_X \Big[\sum \big( b(x, X_i) - \mathbb{E}_X [ b(x, X_i)] \big) \Big]^2 = \\

= \frac{1}{k^2} \mathbb{E}_X \Big[\sum \big( b(x, X_i) - \mathbb{E}_X [ b(x, X_i)] \big)^2 + \sum_{k_1 \neq k_2} \big( b(x, X_{k_1}) - \mathbb{E}_X [ b(x, X_{k_1})] \big)\big( b(x, X_{k_2}) - \mathbb{E}_X [ b(x, X_{k_2})] \big) \Big] = \\

= \frac{1}{k^2} \sum \mathbb{V}_X[b(x, X_i)]  + \frac{1}{k^2} \sum_{k_1 \neq k_2} \text{cov}(b(x, X_{k_1}), b(x, X_{k_2})) \approx \frac{1}{k^2} \sum \mathbb{V}_X[b(x, X_i)] \approx \frac{1}{k}\mathbb{V}_X[b(x, X)]
$$

4. Random Forest
- need to make basic trees uncorrelated
- for each tree use $d < D$ features
    * $D/3$ for classification
    * $\sqrt{D}$ for regression
- basic trees need to have low bias $\rightarrow$ high trees

5. Feature Importance
- same as decision trees, average over all trees
