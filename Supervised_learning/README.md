# 04. Supervised Learning

## Concept

Learning a mapping from inputs (features) to known outputs (labels), using
labeled training examples. Two flavors:

- **Regression** — predict a continuous number (e.g. house price).
- **Classification** — predict a category (e.g. spam / not spam).

## Algorithms covered

| Algorithm | Type | Idea |
|---|---|---|
| Linear Regression | Regression | Fit a straight line minimizing squared error |
| Logistic Regression | Classification | Linear Regression + sigmoid, outputs a probability |
| Decision Tree | Both | Series of if/else splits on features |

Each file implements the algorithm **from scratch** (so you understand the
mechanics), then shows the equivalent one-liner using scikit-learn (what
you'd actually use in practice).

## Files

- `01_linear_regression.py`
- `02_logistic_regression.py`
- `03_decision_trees.py`
