class Flattening:
    def forward(self, x):
        # Save the input shape for the backward pass
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, delta):
        # Reshape the delta to the original input shape
        return delta.reshape(self.input_shape)
