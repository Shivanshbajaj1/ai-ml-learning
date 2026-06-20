"""
04. Logistic Regression — from scratch, then scikit-learn.
Install: pip install numpy scikit-learn
Run directly: python 02_logistic_regression.py
"""
import numpy as np


def sigmoid(z):
    """Squashes any real number into (0, 1) — interpretable as a probability."""
    return 1 / (1 + np.exp(-z))


def logistic_regression_from_scratch(X, y, learning_rate=0.1, epochs=1000):
    """
    Binary classification: predict probability that y=1, using
    sigmoid(w*X + b). Same gradient descent pattern as linear regression,
    just with a different prediction function and loss (log loss).
    """
    n = len(X)
    w, b = 0.0, 0.0

    for epoch in range(epochs):
        z = w * X + b
        y_pred = sigmoid(z)              # predicted probability
        error = y_pred - y

        dw = (1 / n) * np.sum(error * X)
        db = (1 / n) * np.sum(error)

        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 200 == 0:
            # log loss (cross-entropy) — clip to avoid log(0)
            eps = 1e-10
            loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
            print(f"  epoch {epoch:4d}  loss={loss:.4f}")

    return w, b


def predict(X, w, b, threshold=0.5):
    probs = sigmoid(w * X + b)
    return (probs >= threshold).astype(int)


def logistic_regression_sklearn(X, y):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X.reshape(-1, 1), y)
    return model


if __name__ == "__main__":
    # Toy dataset: hours studied -> pass (1) / fail (0)
    np.random.seed(42)
    hours_studied = np.array([1, 2, 2.5, 3, 4, 4.5, 5, 6, 7, 8])
    passed = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

    print("From scratch:")
    w, b = logistic_regression_from_scratch(hours_studied, passed)
    preds = predict(hours_studied, w, b)
    accuracy = np.mean(preds == passed)
    print(f"  final accuracy: {accuracy:.2%}")

    print("\nscikit-learn:")
    model = logistic_regression_sklearn(hours_studied, passed)
    sklearn_preds = model.predict(hours_studied.reshape(-1, 1))
    print(f"  accuracy: {np.mean(sklearn_preds == passed):.2%}")
