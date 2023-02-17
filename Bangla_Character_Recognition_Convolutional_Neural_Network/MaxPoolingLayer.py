import numpy as np

class MaxPooling:
    def __init__(self, filter_dim, stride):
        self.filter_dim = filter_dim
        self.stride = stride

    def forward(self, x):
        # Save the input shape for the backward pass
        self.input_shape = x.shape
        n_h, n_w, n_c = x.shape
        # Initialize the output feature map
        feature_map = np.zeros((n_h // self.stride, n_w // self.stride, n_c))
        # Initialize the indices of the maximum values in the input feature map
        self.max_indices = np.zeros((n_h // self.stride, n_w // self.stride, n_c))
        # Loop over every channel in the output feature map
        for c in range(n_c):
            # Loop over every spatial location in the output feature map
            for h in range(0, n_h, self.stride):
                for w in range(0, n_w, self.stride):
                    # Extract the convolution window from the input image
                    convolution_window = x[h:h + self.filter_dim, w:w + self.filter_dim, c]
                    # Find the maximum value in the convolution window
                    feature_map[h // self.stride, w // self.stride, c] = np.max(convolution_window)
                    # Save the index of the maximum value in the convolution window
                    self.max_indices[h // self.stride, w // self.stride, c] = np.argmax(convolution_window)
        return feature_map

    def backward(self, delta):
        # Initialize the delta for the previous layer
        prev_delta = np.zeros(self.input_shape)
        n_h, n_w, n_c = self.input_shape
        # Loop over every channel in the output feature map
        for c in range(n_c):
            # Loop over every spatial location in the output feature map
            for h in range(0, n_h, self.stride):
                for w in range(0, n_w, self.stride):
                    # Extract the convolution window from the input image
                    convolution_window = prev_delta[h:h + self.filter_dim, w:w + self.filter_dim, c]
                    # Find the maximum value in the convolution window
                    max_index = int(self.max_indices[h // self.stride, w // self.stride, c])
                    # Set the maximum value in the convolution window to the delta of the current layer
                    convolution_window[max_index] = delta[h // self.stride, w // self.stride, c]
        return prev_delta
