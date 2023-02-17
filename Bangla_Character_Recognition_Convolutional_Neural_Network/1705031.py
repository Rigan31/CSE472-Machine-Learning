import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import cv2


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



class ReLU:
    def forward(self, x):
        self.input_x = x
        return np.maximum(x, 0)

    def backward(self, delta):
        # Compute the gradient with respect to the input of the ReLU function
        return delta * (self.input_x > 0)


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


class Flattening:
    def forward(self, x):
        # Save the input shape for the backward pass
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, delta):
        # Reshape the delta to the original input shape
        return delta.reshape(self.input_shape)


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


class ConvoModel:

    def __int__(self, learning_rate=0.01):
        self.input_channels = 1
        self.output_channels = 6
        self.filter_dim = 12
        self.stride = 3
        self.padding = 1
        self.learning_rate = learning_rate

        self.convolution = Convolution(self.input_channels, self.output_channels, self.filter_dim, self.stride, self.padding)
        self.relu = ReLU()
        self.max_pooling = MaxPooling(psool_size=2, stride=2)
        self.flatten = Flattening()
        self.fully_connected = FullyConnected(output_units=10, learning_rate=self.learning_rate)
        self.softmax = Softmax()


    def forward(self, x):
        x = self.convolution.forward(x)
        x = self.relu.forward(x)
        x = self.max_pooling.forward(x)
        x = self.flatten.forward(x)
        x = self.fully_connected.forward(x)
        x = self.softmax.forward(x)

        print("ConvoModel forward: ", x.shape)
        return x

    def backward(self, delta):
        delta = self.softmax.backward(delta)
        delta = self.fully_connected.backward(delta)
        delta = self.flatten.backward(delta)
        delta = self.max_pooling.backward(delta)
        delta = self.relu.backward(delta)
        delta = self.convolution.backward(delta)

        return delta


def make_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def load_images(data_folder):
    images = []
    for file_name in tqdm(os.listdir(data_folder)):
        image = cv2.imread(os.path.join(data_folder, file_name))
        if image is not None:
            gray_image = make_grayscale(image)
            images.append(gray_image)
    return images


def get_maximum_size_of_images(images):
    max_height = 0
    max_width = 0
    for image in images:
        height, width = image.shape
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width

    return max_height, max_width


def preprocess_images(images, size):
    preprocess_images_list = []
    for images in tqdm(images):
        image = cv2.resize(images, size)
        image = np.array(image)/255
        preprocess_images_list.append(image)

    return preprocess_images_list


def load_csv(csv_path):
    csv_data = pd.read_csv(csv_path)
    return csv_data


def get_labels(csv_data, label_name):
    labels = csv_data[label_name]
    return labels


def get_one_hot_encoding(labels):
    one_hot_encoding = pd.get_dummies(labels)
    return one_hot_encoding


def train_test_data():
    image_path = './dataset/NumtaDB_with_aug/training-b//'
    csv_path = './dataset/NumtaDB_with_aug/training-b.csv'

    images = load_images(image_path)
    shape = get_maximum_size_of_images(images)
    print(shape)
    max_height, max_width = shape
    print(max_height, max_width)
    preprocess_images_list = preprocess_images(images, (max_height, max_width))
    csv_data = load_csv(csv_path)
    labels = get_labels(csv_data, 'digit')
    one_hot_encoding = get_one_hot_encoding(labels)
    train_x, train_y = preprocess_images_list, labels
    train_x = np.array(train_x)
    print(train_x.shape)

    return train_x, train_y


if __name__ == '__main__':
    train_x, train_y = train_test_data()
    model = ConvoModel()
    model.forward(train_x)
    model.backward(train_x)