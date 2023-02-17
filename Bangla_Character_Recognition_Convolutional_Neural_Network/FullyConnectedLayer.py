import numpy as np


class FullyConnected:
    def __init__(self, out_units):
        self.out_units = out_units
        self.weights = None
        self.bias = None

    def forward(self, x):
        if self.weights is None:
            self.weights = np.random.randn(x.shape[1], self.out_units) * 0.1
            self.bias = np.zeros((1, self.out_units))
        return np.dot(x, self.weights) + self.bias

    def backward(self, delta, lr):
        # Update the weights and bias
        self.weights -= lr * np.dot(delta.T, self.weights)
        self.bias -= lr * np.mean(delta, axis=0)

        # Calculate the delta for the previous layer
        prev_delta = np.dot(delta, self.weights.T)
        return prev_delta
