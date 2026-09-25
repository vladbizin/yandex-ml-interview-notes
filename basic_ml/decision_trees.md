# Decision Tree
1. Definition
- node: predicate function
- leaf: constant prediction
- instant drawback - non differentiable function

2. Decision Stump
- Before split (leaf)
$$
H(X) = \frac{1}{|X|} \sum L(y, c)
$$
- After split (stump)
$$
H(X) = \frac{1}{|X|} (\sum_{X_l} L(y, c_l) + \sum_{X_r} L(y, c_r)) = \frac{|X_l|}{|X|}H_l + \frac{|X_r|}{|X|}H_r
$$
- Best split: find predicate function such that
$$
\frac{1}{|X|} \sum L(y, c) - \frac{|X_l|}{|X|}H_l + \frac{|X_r|}{|X|}H_r \rightarrow \max
$$
- complexity: O(N^2D) - for each feature value ($N$ values) calculate $H$ decrease (each in $O(ND)$ time)

3. Full Decision Tree
- NP problem -> use greedy approach
- Recursive decision stumps until break condition
- complexity: each node considers $O(qD)$ splits, each requries $O(q)$ for $H$ calculation -> $O(N^2D)$ per level -> $O(hN^2D)$ total

4. Loss Functions and Prediction Functions
- MSE -> $H=Variance Decrease$, $c=\bar{y}$
- MAE -> $c=Median(y)$
- Misslcassif -> $H=1-p_c$, $c=\arg\max_i y_i / \sum y_i$
- Brier -> $H　=　\sum p_i(1-p_i)$, $c=[p_i]$
- Bernoulli Maximum Likelihood -> $H=Entropy(p)$, $c=[p_i]$

5. Missing Values
- Training
    * ignore subset $V \in X$ with missing values for that feature
    * choose split without objects with missing values for that feature
    * subset $V$ goes down each decision path with weights $|X_l| / |X|$ and $|X_r| / |X|$
- Inference
    * object $x$ with missing value for split feature
    * go down each of the paths
    * predict $y=\frac{|X_l|}{|X|}y_l + \frac{|X_r|}{|X|}y_r$
 
6. Optimization
- Categorical features
    * given feature cardinality is $M$, there are $2^{M-1}$ splits of type $x^i \in C_r$, $C=C_l \cup C_r$
    * for binary classification / regression task: sort $c_m$ according to mean target , work with categorical features as with float
    * such split would be the most optimal according to Gini / Entropy or MSE criteria.

- Dynamic programming split selection
    * naive approach: $O(N^2D)$ - $O(ND)$ possible splits, each requires $O(N)$ to calculate impurity decrease ID (also called information gain IG)
    * sort objects by feature value
    * iterate over sorted values, use DP to recalculate IG
    * optimized complexity: sort $D$ features in $O(N \log N)$ each, one liner IG calculatin per feature: $O(ND)$, total is $O(D N \log N + ND) = O(D N \log N)$ vs $O(N^2D)$ naive

- Counting Sort
    * split features into $b$ bins, each bin has approximately the same amount of objects, say 10 bins with 10% each, complexity is $O(N \log N)$
    * now you can use counting sort to sort in $O(N)$ time
    * final complexity for split selection is $O(ND)$
    * total complexity is $O(D N \log N)$ for bin calculation and $O(ND)$ for each level split, resulting in $O(D N \log N + hND)$

7. Regularization
- height
- max leaves
- min objects in leaf
- min IG perentage
- etc

8. Feature Importance
- total IG for that feature
- number of objects splitting on that feature
- biased toward high cardinality categorical features

9. First Implementations
- CART - first regression tree
- ID3 - first tree to use Entropy for ingformation gain
