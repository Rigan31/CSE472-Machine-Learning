import numpy as np

class ReLU:
    def forward(self, x):
        self.input_x = x
        return np.maximum(x, 0)

    def backward(self, delta):
        # Compute the gradient with respect to the input of the ReLU function
        return delta * (self.input_x > 0)
