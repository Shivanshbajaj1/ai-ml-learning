"""
01. NumPy Basics
Install: pip install numpy
Run directly: python 01_numpy_basics.py
"""
import numpy as np


def array_creation():
    a = np.array([1, 2, 3, 4, 5])
    zeros = np.zeros((2, 3))         # 2x3 matrix of zeros
    ones = np.ones((3, 3))           # 3x3 matrix of ones
    identity = np.eye(3)             # 3x3 identity matrix
    rng = np.arange(0, 10, 2)        # like Python range, but returns an array
    lin = np.linspace(0, 1, 5)       # 5 evenly spaced values from 0 to 1

    print("array:", a)
    print("zeros:\n", zeros)
    print("identity:\n", identity)
    print("arange:", rng)
    print("linspace:", lin)


def vectorized_operations():
    """
    The core NumPy idea: operate on entire arrays at once instead of
    Python for-loops. Much faster (NumPy runs in compiled C under the hood).
    """
    a = np.array([1, 2, 3, 4])
    b = np.array([10, 20, 30, 40])

    print("a + b:", a + b)            # element-wise, no loop needed
    print("a * 2:", a * 2)
    print("a ** 2:", a ** 2)
    print("dot product:", np.dot(a, b))
    print("mean:", a.mean(), "| std:", a.std(), "| sum:", a.sum())


def broadcasting():
    """
    Broadcasting lets NumPy apply operations between arrays of different
    (but compatible) shapes, without manually looping or reshaping.
    """
    matrix = np.array([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)
    row = np.array([10, 20, 30])                # shape (3,)

    print("matrix + row (broadcast):\n", matrix + row)
    print("matrix shape:", matrix.shape)


def indexing_and_slicing():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print("element [1,2]:", arr[1, 2])
    print("row 0:", arr[0])
    print("column 1:", arr[:, 1])
    print("sub-matrix [0:2, 0:2]:\n", arr[0:2, 0:2])
    print("boolean mask (arr > 5):", arr[arr > 5])


if __name__ == "__main__":
    array_creation()
    print()
    vectorized_operations()
    print()
    broadcasting()
    print()
    indexing_and_slicing()
