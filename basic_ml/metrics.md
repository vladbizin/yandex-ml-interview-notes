# Metrics
1. Regression
- MSE: optimizes mean
$$
\sum_i (y_i-f(x_i))^2
$$
- MAE: optimizes median
$$
\sum_i |y_i-f(x_i)|
$$
- MAPE: where order of magnitude should not matter
$$
\sum_i \frac{|y_i-f(x_i)|}{|y_i|}
$$
- SMAPE: if target can be equal to zero
$$
\sum_i \frac{|y_i-f(x_i)|}{|y_i+f(x_i)|/2}
$$
- $R^2$: how much trained algorithm is better than just predicting mean
$$
1 - \frac{\sum_i (y_i - f(x_i)^2)}{\sum_i (y_i - \bar{y})^2}
$$

2. Classification
- Accuracy
$$
\frac{TP + TN}{TP + TN + FP + FN}
$$
- Precision
$$
\frac{TP}{TP + FP}
$$
- Recall
$$
\frac{TP}{TP + FN}
$$

3. ROC
- Curve with TPR on y-axis, FPR on x-axis
$$
TPR = \frac{TP}{TP + FN} \\

FPR = \frac{FP}{FP + TN}
$$
- non-decreasing function

3. ROC-AUC
- area under ROC-curve
- ratio of correctly-ordered pairs
$$
\frac{\sum_{i \neq j} [f(x_i) < f(x_j)] [y_i < y_j]}{\sum_{i \neq j} [y_i < y_j]}
$$
- highly biased towards 1 for imbalanced dataset (most 1, few 0)
- roc-auc of a random classifier = 0.5

6. Pr-Re Curve
- recall on x-asis, precision on y-axis
- not monotonous
- for minimal classification threshold: precision = target rate
- random classifier: precision is constant = target rate, recall is monotonous in $[0, 1]$
