# Gradient Boosting

1. Intuition (MSE)
- target $y$, alogrithm $a_0(x) = b_0(x)$
- train $b_1$ to predict $y-a_0(x)$, so that $a_1(x) = a_0(x) + b_1(x) = b_0(x) + b_1(x) \approx a_0(x) + y - a_0(x) \approx y$
- in general for $a_i(x) = \sum_k^i b_k(x)$ we train $b_{i+1}$ to predict $y-a_i(x)$
- in case of MSE Loss $L(y, a) = (y-a(x))^2$ $y-a_i(x)$ happens to be $- \nabla L(y, a=a_i)$

2. Generalization
- at step $i$ we train basic algorithm $b_{i+1}$
$$
b_{i+1} = \arg \min_b L(y, a_i + b)
$$
- Taylor Expansion of $L$
$$
L(y, a_i + b) = L(y, a_i) + b \frac{\partial L}{\partial a}\bigg|_{a-a_i} \rightarrow \min \Rightarrow b_{i+1} \propto -\frac{\partial L}{\partial a}\bigg|_{a-a_i}
$$
- generalizes to any differentiable Loss function
- always a regression task: predict float (e.g. logit in case of binary cross entrpopy loss)

3. Interpretation
- given a dataset of $N$ points, each iteration of basic algorithm training is a gradient descent in space of predicted values
$$
\Big( a_{i+1}(x_0), a_{i+1}(x_1), ..., a_{i+1}(x_N) \Big) =
\Big( (a_i + \eta b_{i+1})(x_0), (a_i + \eta b_{i+1})(x_1), ..., (a_i + \eta b_{i+1})(x_N) \Big)
$$

4. Basic Algorithms
- high bias
- low variance
- weak learners (low trees)

5. Implementations
- CatBoost
    * encodes categorical features using target statistics
    * symmetric balanced trees: each level splits on the same feature, tree is complete - speeds up prediction times, acts as regularizer
- XGBoost
    * second order derivatives - less iterations for convergence
    * L2 and L2 regularizations in objective functions
    * handles missing values
        * ignore objects with missing values
        * tests sending all ojbects with missing values to left/right. chooses direction that maximizes gain
        * at inference follow lerned path for missing value
        * 
    * sorts and splits data into blocks for parallel learning
- LightGBM
    * Leaf-Wise Tree Growth
    * Histogram-Based Splitting
    * Gradient-Based One-Side Sampling (GOSS)
        * drops objects with small gradients (already well-trained)

6. Feature Importance
- same as for single tree, but averaged
- some built-in algorithms
