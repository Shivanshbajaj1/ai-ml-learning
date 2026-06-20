# 07. Neural Networks

## Concept

A neural network is layers of simple units ("neurons") that each compute a
weighted sum of inputs, pass it through a non-linear **activation function**,
and feed the result to the next layer. Stacking these layers lets the network
approximate complex, non-linear functions that plain linear/logistic
regression can't.

## Key pieces

- **Forward pass**: compute predictions by pushing input through each layer.
- **Loss function**: measures how wrong the prediction was.
- **Backpropagation**: the chain rule, applied layer by layer, to compute how
  much each weight contributed to the error.
- **Gradient descent**: update every weight a small step opposite its gradient.

This file builds a tiny 2-layer network **using only NumPy** — no PyTorch/
TensorFlow — so you see every piece of the math directly. Once this clicks,
moving to a framework is just "the same ideas, with autograd doing the
calculus for you."

## Files

- `01_simple_neural_network.py` — a 2-layer network trained on XOR (a problem
  that plain linear models *cannot* solve, which is exactly why hidden
  layers + non-linearity matter)
