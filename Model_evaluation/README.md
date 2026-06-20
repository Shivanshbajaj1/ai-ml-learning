# 06. Model Evaluation

## Why "accuracy" alone can lie to you

If 95% of emails are not spam, a model that always predicts "not spam" gets
95% accuracy while being useless. You need metrics that account for this.

## Classification metrics

| Metric | Formula | What it tells you |
|---|---|---|
| Accuracy | (TP+TN) / total | Overall correctness — misleading on imbalanced data |
| Precision | TP / (TP+FP) | Of predicted positives, how many were right? |
| Recall | TP / (TP+FN) | Of actual positives, how many did we catch? |
| F1 Score | 2·(P·R)/(P+R) | Harmonic mean of precision & recall — balances both |

TP = True Positive, FP = False Positive, FN = False Negative, TN = True Negative.

## Regression metrics

| Metric | What it tells you |
|---|---|
| MAE (Mean Absolute Error) | Average absolute size of errors, same units as target |
| MSE (Mean Squared Error) | Like MAE but penalizes large errors more |
| R² Score | Proportion of variance explained (1 = perfect, 0 = no better than guessing the mean) |

## Files

- `01_evaluation_metrics.py` — all metrics implemented from scratch +
  scikit-learn equivalents, plus a confusion matrix and cross-validation example
