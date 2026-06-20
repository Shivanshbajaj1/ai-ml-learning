"""
04. Decision Trees — core splitting concept from scratch, then scikit-learn.
Install: pip install numpy scikit-learn
Run directly: python 03_decision_trees.py
"""
import numpy as np


def gini_impurity(labels):
    """
    Measures how 'mixed' a set of labels is. 0 = perfectly pure (all
    same class), higher = more mixed. Decision trees pick splits that
    minimize this after the split.
    """
    if len(labels) == 0:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return 1 - np.sum(probabilities ** 2)


def best_split(X, y):
    """
    Try every feature and every threshold, find the split that produces
    the lowest weighted Gini impurity in the two resulting groups.
    This is the core operation repeated at every node of a decision tree.
    """
    best_gini = float("inf")
    best_feature, best_threshold = None, None
    n_features = X.shape[1]

    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            left_gini = gini_impurity(y[left_mask])
            right_gini = gini_impurity(y[right_mask])
            weighted_gini = (
                left_mask.sum() * left_gini + right_mask.sum() * right_gini
            ) / len(y)

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature, best_threshold = feature_idx, threshold

    return best_feature, best_threshold, best_gini


def decision_tree_sklearn(X, y):
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    return model


if __name__ == "__main__":
    # Toy dataset: [hours_studied, hours_slept] -> pass (1) / fail (0)
    X = np.array([
        [1, 8], [2, 7], [3, 6], [4, 5],
        [5, 6], [6, 7], [7, 5], [8, 4],
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])

    print("Root node Gini impurity:", gini_impurity(y))

    feature_idx, threshold, gini = best_split(X, y)
    feature_name = "hours_studied" if feature_idx == 0 else "hours_slept"
    print(f"Best first split: {feature_name} <= {threshold}  (gini={gini:.4f})")

    print("\nscikit-learn DecisionTreeClassifier:")
    model = decision_tree_sklearn(X, y)
    print("  predictions:", model.predict(X))
    print("  feature importances:", model.feature_importances_)
