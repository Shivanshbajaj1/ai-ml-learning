"""
04. Linear Regression — from scratch with gradient descent, then scikit-learn.
Install: pip install numpy scikit-learn
Run directly: python 01_linear_regression.py
"""
import numpy as np


def linear_regression_from_scratch(X, y, learning_rate=0.01, epochs=1000):
    """
    Fit y = w*X + b using gradient descent.
    This is THE foundational training loop pattern used across all of ML:
    predict -> measure error -> compute gradient -> update parameters -> repeat.
    """
    n = len(X)
    w, b = 0.0, 0.0   # start with arbitrary initial parameters

    for epoch in range(epochs):
        y_pred = w * X + b                      # 1. predict
        error = y_pred - y                       # 2. measure error

        # 3. gradient: how much does the loss (mean squared error) change
        #    with respect to each parameter? (calculus, but the formulas
        #    below are the result — you don't need to re-derive them)
        dw = (2 / n) * np.sum(error * X)
        db = (2 / n) * np.sum(error)

        # 4. update parameters, stepping AGAINST the gradient
        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 200 == 0:
            mse = np.mean(error ** 2)
            print(f"  epoch {epoch:4d}  mse={mse:.4f}  w={w:.4f}  b={b:.4f}")

    return w, b


def linear_regression_sklearn(X, y):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    return model.coef_[0], model.intercept_


if __name__ == "__main__":
    # Generate data that roughly follows y = 3x + 5, plus noise
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = 3 * X + 5 + np.random.randn(50) * 2

    print("From scratch (gradient descent):")
    w, b = linear_regression_from_scratch(X, y)
    print(f"  final: w={w:.4f}, b={b:.4f}  (true values: w=3, b=5)")

    print("\nscikit-learn:")
    w_sklearn, b_sklearn = linear_regression_sklearn(X, y)
    print(f"  w={w_sklearn:.4f}, b={b_sklearn:.4f}")
