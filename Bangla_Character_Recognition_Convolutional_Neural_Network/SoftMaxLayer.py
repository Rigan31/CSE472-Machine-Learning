import numpy as np

class Softmax:
    def forward(self, x):
        # Shift the input to avoid numeric instability
        x -= np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, delta, y):
        # Calculate the delta for the previous layer
        prev_delta = delta * (y - delta)
        return prev_delta
