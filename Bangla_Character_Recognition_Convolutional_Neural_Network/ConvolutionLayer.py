import numpy as np


class Convolution:
    def __init__(self, output_channels, filter_dim, stride, padding):
        self.output_channels = output_channels
        self.filter_dim = filter_dim
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(output_channels, filter_dim, filter_dim, 1)
        self.bias = np.zeros((output_channels, 1))

    def forward(self, input_image):
        self.input_shape = input_image.shape
        n_img, n_h, n_w, n_c = input_image.shape
        # Add padding to the input image
        input_image = np.pad(input_image, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
        # Initialize the output feature map
        feature_map = np.zeros((n_h // self.stride, n_w // self.stride, self.output_channels))
        # Loop over every channel in the output feature map
        for c in range(self.output_channels):
            # Loop over every spatial location in the output feature map
            for h in range(0, n_h, self.stride):
                for w in range(0, n_w, self.stride):
                    # Extract the convolution window from the input image
                    convolution_window = input_image[h:h + self.filter_dim, w:w + self.filter_dim, :]
                    # Compute the dot product between the convolution window and the weights
                    feature_map[h // self.stride, w // self.stride, c] = np.sum(convolution_window * self.weights[c]) + self.bias[c]
        return feature_map
