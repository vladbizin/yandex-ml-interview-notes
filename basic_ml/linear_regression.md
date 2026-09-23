# Linear Regression
1. Problem Statement
$$
L = \| y - Xw \|_2^2 \rightarrow \min_w
$$

2. Exact Solution 
$$
\nabla_w L = 2 X^T (y - Xw) = 0 \\
X^Ty - X^TXw = 0 \\
X^TXw = X^Ty \\
w = (X^TX)^{-1}X^Ty
$$
- complexity $O(D^2N+D^3)$

3. Prove positive difinite
$$
\nabla_w^2 L = 2X^TX \\

w^T(X^TX)w = (Xw)^T(Xw) >=0
$$

4. Gradient descent
$$
\nabla_w L(w,X,y) = \frac{1}{N} \sum_i \nabla_{w} L(w,x_i,y_i) \approx \frac{1}{B} \sum_t \nabla_{w} L(w,x_{i_t},y_{i_t}) \\

w_{k+1} = w_k - \alpha_k \nabla_w L(w_k,X,Y)
$$
- non-biased Monte-Carlo estimate
- complexity of each epoch is $O(ND)$, but weights are updated $N/B$ as often

5. Optimal step size
$$
\alpha_k = \min_\alpha L(w_k - \alpha_k \nabla_w L(w_k,X,Y), X, y)
$$

6. Second order
- Second Order
    * Newton method
    * $f(x + \Delta x) \sim f(x) + \langle \nabla f(x), \Delta x \rangle + \frac{1}{2} \langle H(x) \Delta x, \Delta x \rangle$
    * $\Delta x = H^{-1}(x) \nabla f(x)$
    * $x_{k+1} = x_k - \alpha H_k^{-1} \nabla f(x_k)$

- Quasi-Newthon methods
    * Secant method
    * BFGS and L-BFGS

7. Gradient Descent Modifications
- GD
    * direction of fastest local increase of a function.
        * Taylor expansion for $\|h\|=1$: $f(x+\alpha h) = f(x) + \alpha \langle \nabla f(x), h \rangle + o(\alpha)$
        * We want to find minimal value: $f(x) + \alpha \langle \nabla f(x), h \rangle < f(x)$
        * Cauchy–Schwarz inequality: $\langle \nabla f(x), h \rangle \geq \| \nabla f(x) \|_2 \| h \|_2 = \| \nabla f(x) \|_2$
        * where $\langle \nabla f(x), h \rangle$ takes minimal value when $h \sim - \nabla f(x) = - \frac{\nabla f(x)}{\| \nabla f(x) \|_2}$
    * hence $x_{k+1} = x_k - \alpha \nabla f(x_k)$
    * fastest descent: $\alpha_k = \arg \min_\alpha f \big( x_k - \alpha \nabla f(x_k) \big)$

- SGD
    * $f(x) = \sum_i L(x, y_i)$
    * $f(x) = \mathbb{E} \big[ L(x, \xi) \big], \xi \sim y$
    * $\nabla f(x) = \mathbb{E} \big[ \nabla L(x, y) \big]$
    * unbiased Monte-Carlo estimate: $\tilde{\nabla} f(x) = \frac{1}{B} \sum_{i=1}^{B} \nabla L(x, \xi_i)$

- Momentum
    * $v_{k+1} = \beta_k v_k - \alpha_k \nabla f(x_k)$
    * $x_{k+1} = x_k + v_{k+1}$

- Nesterov Momentum
    * $v_{k+1} = \beta_k v_k - \alpha_k \nabla f(x_k + \beta_k v_k)$
    * $x_{k+1} = x_k + v_{k+1}$

- AdaGrad
    * $G_{k+1} = G_k + \big( \nabla f (x_k) \big)^2$
    * $x_{k+1} = x_k - \frac{\alpha}{\sqrt{G_{k+1} + \varepsilon}} \nabla f(x_k)$

- RMSPROP
    * $G_{k+1} = \gamma  G_k + (1 - \gamma) \big( \nabla f (x_k) \big)^2$
    * $x_{k+1} = x_k - \frac{\alpha}{\sqrt{G_{k+1} + \varepsilon}} \nabla f(x_k)$

- Adam
    * $v_{k+1} = \beta_1 v_k + (1-\beta_1) \nabla f(x_k)$
    * $G_{k+1} = \beta_2  G_k + (1 - \beta_2) \big( \nabla f (x_k) \big)^2$
    * $x_{k+1} = x_k - \frac{\alpha}{\sqrt{G_{k+1} + \varepsilon}} v_{k+1}$

- AdamW
    * $v_{k+1} = \beta_1 v_k + (1-\beta_1) \nabla f(x_k)$
    * $G_{k+1} = \beta_2  G_k + (1 - \beta_2) \big( \nabla f (x_k) \big)^2$
    * $x_{k+1} = x_k - (\frac{\alpha}{\sqrt{G_{k+1} + \varepsilon}} v_{k+1} + \lambda x_k)$

8. Why regularization is needed  
- Consider X is degenerate (features have linear dependency), then $\exist \nu: X\nu = 0$
- Consider $w$ to be a solution of the problem: $w:X^TXw=X^Ty$
- Then sum $w+\nu$ is also a solution:
$$
X^TX(w+\nu) = X^T(Xw + X\nu) = X^TXw=X^Ty
$$
- since $\nu$ can be arbitrary large, the solution becomes norm-unbounded

9. Regularization types
- L2
    * $L = L + \|w\|_2^2$
    * $w = (X^TX + \lambda I)^{-1}X^Ty$
- L1
    * $L = L + \|w\|_1^1$
    * feature selections due to weight sparsity


10. Feature normalization
- does not change predictions
- makes weights interpretable

11. Loss functions
- MAE
    * $\frac{1}{N} \sum |y_i-\langle w, x_i \rangle|$
    * approximates median of $p(y|x)$
- MAPE
    * $\frac{1}{N} \sum \Big| \frac{y_i-\langle w, x_i \rangle}{y_i} \Big|$
    * for variable scale targets
- Huber
    * $\frac{1}{N} \sum h_d(y_i-\langle w, x_i \rangle)$
    * $h_d(x)=x^2/2, |x| \leq 2$
    * $h_d(x)= d(|x|-d/2), |x|>d$

12. Metrics
- MSE
    * $MSE = \frac{1}{N} \sum (y_i - \langle w, x_i \rangle)^2$
- RMSE
    * $RMSE = \sqrt{MSE}$
    * cast units from squared target to target
- R2
    * $R^2 = 1 - \frac{\sum (y_i - \langle w, x_i \rangle)^2}{\sum (y_i- \bar{y})^2}$
    * explained variance, normalized MSE
- MAE
    * $MAE = \frac{1}{N} \sum |y_i - \langle w, x_i \rangle|$
    * suitable for datasets with lots of outliers
- MAPE
    * $MAPE = \frac{1}{N} \sum \Big| \frac{y_i-\langle w, x_i \rangle}{y_i} \Big|$
    * suitable for datasets with variable scale targets
- SMAPE
    * $SMAPE = \frac{1}{N} \sum \frac{|y_i-\langle w, x_i \rangle|}{(|y_i|+|\langle w, x_i \rangle|) / 2}$
    * suitable when MAPE is preferred, but targets can be equal to 0
- RMSLE
    * $RMSLE = \sqrt{\frac{1}{N} \sum \log{\frac{y_i+c}{\langle w, x_i \rangle+c}}}$
    * suitable for cases where it is most important to predict same scale as target value, rather than exact same value
    * similair to log-scaling your targets first for training
14. MLE estimate
$$
y = \langle w, x \rangle + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2) \\

y \sim \mathcal{N}(\langle w, x \rangle, \sigma^2) \\

ML = \prod p(y_i|x_i, w) = \prod \mathcal{N}(\langle w, x_i \rangle, \sigma^2) \\

\log{ML} = \frac{1}{N} \sum (y_i - \langle w, x_i \rangle)^2 + C \\

\arg \max \log{ML} = \arg \max \frac{1}{N} \sum (y_i - \langle w, x_i \rangle)^2
$$
- Ordinary Least Squares estimator is the Maximal Likelihood Estimator
- if $\epsilon \sim \text{Laplace}$, then MLE is the estimator minimizing MAE
- OLS estimator is the Best Linear Unbiased Estimator (Gauss-Markov Theorem)
    * Best: has the smallest variance in class (among all linear unbiased estimators)
    * Unbiased: $\mathbb{E}[\tilde{w}]=w$
    * $\tilde{w} \xrightarrow{\mathbb{P}} w$
