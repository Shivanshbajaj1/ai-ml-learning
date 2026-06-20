"""
07. A minimal neural network from scratch, in pure NumPy.
Trains on XOR — a classic example of a problem that is NOT linearly separable,
which is exactly why we need a hidden layer (a single-layer / logistic
regression model provably cannot solve XOR).
Run directly: python 01_simple_neural_network.py
"""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    """Derivative of sigmoid, expressed in terms of its OWN output `a` (a neat trick)."""
    return a * (1 - a)


class SimpleNeuralNetwork:
    """A 2-layer network: input -> hidden layer -> output."""

    def __init__(self, input_size, hidden_size, output_size, seed=42):
        rng = np.random.default_rng(seed)
        # Small random initial weights — break symmetry between neurons
        self.W1 = rng.standard_normal((input_size, hidden_size)) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.standard_normal((hidden_size, output_size)) * 0.5
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """Push input through both layers, caching intermediate values for backprop."""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)              # hidden layer activation
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)              # output (prediction)
        return self.a2

    def backward(self, X, y, learning_rate):
        """
        Backpropagation: compute how much each weight contributed to the
        error, working backward from the output layer to the input layer
        (the chain rule, applied mechanically).
        """
        n = X.shape[0]

        # --- output layer gradients ---
        d_a2 = self.a2 - y                                  # derivative of loss w.r.t. output
        d_z2 = d_a2 * sigmoid_derivative(self.a2)
        d_W2 = self.a1.T @ d_z2 / n
        d_b2 = np.sum(d_z2, axis=0, keepdims=True) / n

        # --- hidden layer gradients (error flows backward through W2) ---
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * sigmoid_derivative(self.a1)
        d_W1 = X.T @ d_z1 / n
        d_b1 = np.sum(d_z1, axis=0, keepdims=True) / n

        # --- gradient descent update ---
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1

    def train(self, X, y, epochs=5000, learning_rate=0.5):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, learning_rate)

            if epoch % 1000 == 0:
                loss = np.mean((y_pred - y) ** 2)
                print(f"  epoch {epoch:5d}  loss={loss:.4f}")

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)


if __name__ == "__main__":
    # XOR truth table: a classic non-linearly-separable problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    print("Training on XOR:")
    nn.train(X, y)

    print("\nPredictions after training:")
    predictions = nn.predict(X)
    for inputs, true_val, pred in zip(X, y, predictions):
        print(f"  input={inputs}  true={true_val[0]}  predicted={pred[0]}")
