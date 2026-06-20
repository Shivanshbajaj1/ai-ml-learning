"""
06. Model Evaluation Metrics — from scratch, then scikit-learn.
Install: pip install numpy scikit-learn
Run directly: python 01_evaluation_metrics.py
"""
import numpy as np


# ---------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------

def confusion_matrix_counts(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_counts(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def evaluate_classification_demo():
    # Imbalanced example: 9 negatives, 1 positive, deliberately mostly-wrong model
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # never predicts the positive class

    print("Imbalanced data example (model NEVER predicts the rare class):")
    print(f"  accuracy:  {accuracy(y_true, y_pred):.2f}   <- looks great...")
    print(f"  precision: {precision(y_true, y_pred):.2f}")
    print(f"  recall:    {recall(y_true, y_pred):.2f}   <- ...but this reveals it's useless")
    print(f"  f1_score:  {f1_score(y_true, y_pred):.2f}")


def evaluate_regression_demo():
    y_true = np.array([3.0, -0.5, 2.0, 7.0, 4.2])
    y_pred = np.array([2.5, 0.0, 2.1, 7.8, 4.0])

    print("\nRegression metrics:")
    print(f"  MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"  R²:  {r2_score(y_true, y_pred):.4f}")


def sklearn_equivalents():
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score as sk_f1,
        confusion_matrix,
    )

    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])

    print("\nscikit-learn equivalents:")
    print("  confusion_matrix:\n", confusion_matrix(y_true, y_pred))
    print(f"  accuracy:  {accuracy_score(y_true, y_pred):.2f}")
    print(f"  precision: {precision_score(y_true, y_pred):.2f}")
    print(f"  recall:    {recall_score(y_true, y_pred):.2f}")
    print(f"  f1:        {sk_f1(y_true, y_pred):.2f}")


def cross_validation_example():
    """
    A single train/test split can be lucky or unlucky. K-fold cross-validation
    splits the data into k folds, trains/tests k times (each fold as test once),
    and averages the results — a more reliable performance estimate.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    model = LogisticRegression()
    scores = cross_val_score(model, X, y, cv=5)

    print("\n5-fold cross-validation accuracy scores:", np.round(scores, 3))
    print(f"  mean: {scores.mean():.3f}  std: {scores.std():.3f}")


if __name__ == "__main__":
    evaluate_classification_demo()
    evaluate_regression_demo()
    sklearn_equivalents()
    cross_validation_example()
