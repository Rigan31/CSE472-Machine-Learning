import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import cv2
import math
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix


class ConvolutionLayer:
    def __init__(self, num_filters, kernel_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.biases = None
        self.weights_matrix = None
        self.biases_vector = None
        self.u_pad = None

    def __str__(self):
        return f'Conv(filter={self.num_filters}, kernel={self.kernel_size}, stride={self.stride}, padding={self.padding})'

    def forward(self, u):
        num_samples = u.shape[0]
        input_dim = u.shape[1]
        output_dim = math.floor((input_dim - self.kernel_size + 2 * self.padding) / self.stride) + 1
        num_channels = u.shape[3]

        if self.weights is None:
            # ref: https://cs231n.github.io/neural-networks-2/#init
            # ref: https://stats.stackexchange.com/questions/198840/cnn-xavier-weight-initialization
            self.weights = np.random.randn(self.num_filters, self.kernel_size, self.kernel_size,
                                           num_channels) * math.sqrt(
                2 / (self.kernel_size * self.kernel_size * num_channels))
        if self.biases is None:
            # ref: https://cs231n.github.io/neural-networks-2/#init
            self.biases = np.zeros(self.num_filters)

        self.u_pad = np.pad(u, ((0,), (self.padding,), (self.padding,), (0,)), mode='constant')
        v = np.zeros((num_samples, output_dim, output_dim, self.num_filters))

        for k in range(num_samples):
            for l in range(self.num_filters):
                for i in range(output_dim):
                    for j in range(output_dim):
                        v[k, i, j, l] = np.sum(self.u_pad[k, i * self.stride: i * self.stride + self.kernel_size,
                                               j * self.stride: j * self.stride + self.kernel_size, :] * self.weights[
                                                   l]) + self.biases[l]

        return v

    def backward(self, del_v, lr):
        num_samples = del_v.shape[0]
        input_dim = del_v.shape[1]
        input_dim_pad = (input_dim - 1) * self.stride + 1
        output_dim = self.u_pad.shape[1] - 2 * self.padding
        num_channels = self.u_pad.shape[3]

        del_b = np.sum(del_v, axis=(0, 1, 2)) / num_samples
        del_v_sparse = np.zeros((num_samples, input_dim_pad, input_dim_pad, self.num_filters))
        del_v_sparse[:, :: self.stride, :: self.stride, :] = del_v
        weights_prime = np.rot90(np.transpose(self.weights, (3, 1, 2, 0)), 2, axes=(1, 2))
        del_w = np.zeros((self.num_filters, self.kernel_size, self.kernel_size, num_channels))

        for l in range(self.num_filters):
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    del_w[l, i, j, :] = np.mean(np.sum(
                        self.u_pad[:, i: i + input_dim_pad, j: j + input_dim_pad, :] * np.reshape(
                            del_v_sparse[:, :, :, l], del_v_sparse.shape[: 3] + (1,)), axis=(1, 2)), axis=0)

        del_u = np.zeros((num_samples, output_dim, output_dim, num_channels))
        del_v_sparse_pad = np.pad(del_v_sparse, (
        (0,), (self.kernel_size - 1 - self.padding,), (self.kernel_size - 1 - self.padding,), (0,)), mode='constant')

        for k in range(num_samples):
            for l in range(num_channels):
                for i in range(output_dim):
                    for j in range(output_dim):
                        del_u[k, i, j, l] = np.sum(
                            del_v_sparse_pad[k, i: i + self.kernel_size, j: j + self.kernel_size, :] * weights_prime[l])

        self.update_learnable_parameters(del_w, del_b, lr)
        return del_u

    def update_learnable_parameters(self, del_w, del_b, lr):
        self.weights = self.weights - lr * del_w
        self.biases = self.biases - lr * del_b

    def save_learnable_parameters(self):
        self.weights_matrix = np.copy(self.weights)
        self.biases_vector = np.copy(self.biases)

    def set_learnable_parameters(self):
        self.weights = self.weights if self.weights_matrix is None else np.copy(self.weights_matrix)
        self.biases = self.biases if self.biases_vector is None else np.copy(self.biases_vector)


class ReLUActivation:
    def __init__(self):
        self.u = None

    def __str__(self):
        return 'ReLU'

    def forward(self, u):
        self.u = u
        v = np.copy(u)
        v[v < 0] = 0  # applying ReLU activation function
        return v

    def backward(self, del_v, lr):
        del_u = np.copy(self.u)
        del_u[del_u > 0] = 1  # applying sign(x) function for x > 0
        del_u[del_u < 0] = 0  # applying sign(x) function for x < 0
        del_u = del_v * del_u
        return del_u


# Max Pooling Layer Definition
class MaxPoolingLayer:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.u_shape = None
        self.v_map = None

    def __str__(self):
        return f'MaxPool(kernel={self.kernel_size}, stride={self.stride})'

    def forward(self, u):
        self.u_shape = u.shape

        num_samples = u.shape[0]
        input_dim = u.shape[1]
        output_dim = math.floor((input_dim - self.kernel_size) / self.stride) + 1
        num_channels = u.shape[3]

        v = np.zeros((num_samples, output_dim, output_dim, num_channels))
        self.v_map = np.zeros((num_samples, output_dim, output_dim, num_channels)).astype(np.int32)

        for k in range(num_samples):
            for l in range(num_channels):
                for i in range(output_dim):
                    for j in range(output_dim):
                        v[k, i, j, l] = np.max(u[k, i * self.stride: i * self.stride + self.kernel_size,
                                               j * self.stride: j * self.stride + self.kernel_size, l])
                        self.v_map[k, i, j, l] = np.argmax(u[k, i * self.stride: i * self.stride + self.kernel_size,
                                                           j * self.stride: j * self.stride + self.kernel_size, l])

        return v

    def backward(self, del_v, lr):
        del_u = np.zeros(self.u_shape)

        num_samples = del_v.shape[0]
        input_dim = del_v.shape[1]
        num_channels = del_v.shape[3]

        for k in range(num_samples):
            for l in range(num_channels):
                for i in range(input_dim):
                    for j in range(input_dim):
                        position = tuple(sum(pos) for pos in zip((self.v_map[k, i, j, l] // self.kernel_size,
                                                                  self.v_map[k, i, j, l] % self.kernel_size),
                                                                 (i * self.stride, j * self.stride)))
                        del_u[(k,) + position + (l,)] = del_u[(k,) + position + (l,)] + del_v[k, i, j, l]

        return del_u


# Flattening Layer Definition
class FlatteningLayer:
    def __init__(self):
        self.u_shape = None

    def __str__(self):
        return 'Flatten'

    def forward(self, u):
        self.u_shape = u.shape
        v = np.copy(u)
        v = np.reshape(v, (v.shape[0], np.prod(v.shape[1:])))
        v = np.transpose(v)
        return v

    def backward(self, del_v, lr):
        del_u = np.copy(del_v)
        del_u = np.transpose(del_u)
        del_u = np.reshape(del_u, self.u_shape)
        return del_u


# Fully Connected Layer Definition
class FullyConnectedLayer:
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.weights = None
        self.biases = None
        self.weights_matrix = None
        self.biases_vector = None
        self.u = None

    def __str__(self):
        return f'FullyConnected(output_dim={self.output_dim})'

    def forward(self, u):
        self.u = u

        if self.weights is None:
            # ref: https://cs231n.github.io/neural-networks-2/#init
            self.weights = np.random.randn(self.output_dim, u.shape[0]) * math.sqrt(2 / u.shape[0])
        if self.biases is None:
            # ref: https://cs231n.github.io/neural-networks-2/#init
            self.biases = np.zeros((self.output_dim, 1))

        v = self.weights @ u + self.biases
        return v

    def backward(self, del_v, lr):
        del_w = (del_v @ np.transpose(self.u)) / del_v.shape[1]
        del_b = np.reshape(np.mean(del_v, axis=1), (del_v.shape[0], 1))
        del_u = np.transpose(self.weights) @ del_v
        self.update_learnable_parameters(del_w, del_b, lr)
        return del_u

    def update_learnable_parameters(self, del_w, del_b, lr):
        self.weights = self.weights - lr * del_w
        self.biases = self.biases - lr * del_b

    def save_learnable_parameters(self):
        self.weights_matrix = np.copy(self.weights)
        self.biases_vector = np.copy(self.biases)

    def set_learnable_parameters(self):
        self.weights = self.weights if self.weights_matrix is None else np.copy(self.weights_matrix)
        self.biases = self.biases if self.biases_vector is None else np.copy(self.biases_vector)


# Softmax Layer Definition
class SoftmaxLayer:
    def __init__(self):
        pass

    def __str__(self):
        return 'Softmax'

    def forward(self, u):
        v = np.exp(u)
        v = v / np.sum(v, axis=0)
        return v

    def backward(self, del_v, lr):
        del_u = np.copy(del_v)
        return del_u



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ConvoModel:

    def __init__(self, learning_rate=0.001, batch_size=32):
        self.input_channels = 1
        self.output_channels = 6
        self.filter_dim = 5
        self.stride = 1
        self.padding = 1
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.convolution_layer = ConvolutionLayer(self.output_channels, self.filter_dim, self.stride, self.padding)
        self.relu_layer = ReLUActivation()
        self.max_pooling_layer = MaxPoolingLayer(2, 2)
        self.flatten_layer = FlatteningLayer()
        self.fully_connected_layer = FullyConnectedLayer(10)
        self.softmax_layer = SoftmaxLayer()


    def forward(self, x):
        x = self.convolution_layer.forward(x)
        x = self.relu_layer.forward(x)
        x = self.max_pooling_layer.forward(x)
        x = self.flatten_layer.forward(x)
        x = self.fully_connected_layer.forward(x)
        x = self.softmax_layer.forward(x)
        return x

    def backward(self, delta):
        delta = self.softmax_layer.backward(delta, self.learning_rate)
        delta = self.fully_connected_layer.backward(delta, self.learning_rate)
        delta = self.flatten_layer.backward(delta, self.learning_rate)
        delta = self.max_pooling_layer.backward(delta, self.learning_rate)
        delta = self.relu_layer.backward(delta, self.learning_rate)
        delta = self.convolution_layer.backward(delta, self.learning_rate)
        return delta


def make_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def load_images(data_folder):
    images = []
    file_name_list = []
    for file_name in tqdm(os.listdir(data_folder)):
        #print("file_name: ", file_name)
        image = cv2.imread(os.path.join(data_folder, file_name))
        if image is not None:
            gray_image = make_grayscale(image)
            images.append(gray_image)
    # print("image shape")
    # images = np.array(images)[:, None]
    # print(images.shape)
    # print("file_name_list: ", file_name_list)
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
        image = np.array(image) / 255
        preprocess_images_list.append(image)
    mean_image = np.mean(preprocess_images_list)
    std_image = np.std(preprocess_images_list)
    preprocess_images_list = (preprocess_images_list - mean_image) / std_image
    # print("preprocess_images_list shape: ", np.array(preprocess_images_list).shape)

    # mean_image = np.mean(preprocess_images_list)
    # std_image = np.std(preprocess_images_list)
    # demo_image_list = (preprocess_images_list - mean_image) / std_image

    preprocess_images_list = np.array(preprocess_images_list)
    preprocess_images_list = np.expand_dims(preprocess_images_list, axis=-1)
    # print("preprocess_images_list shape: ", preprocess_images_list.shape)
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

    # images = np.array(images)
    # shape = get_maximum_size_of_images(images)
    max_height, max_width = 28, 28
    preprocess_images_list = preprocess_images(images, (max_height, max_width))
    csv_data = load_csv(csv_path)
    labels = get_labels(csv_data, 'digit')
    one_hot_encoding = get_one_hot_encoding(labels)
    # train_x, train_y = preprocess_images_list, labels
    # train_x = np.array(train_x)
    # print(train_x.shape)
    # shuffle the data
    # preprocess_images_list, one_hot_encoding = shuffle(preprocess_images_list, one_hot_encoding)
    preprocess_images_list = np.array(preprocess_images_list)
    train_size = int(len(preprocess_images_list) * 0.8)
    train_x, test_x = preprocess_images_list[0:train_size], preprocess_images_list[
                                                            train_size:len(preprocess_images_list)]
    # train_y, test_y = one_hot_encoding[0:train_size], one_hot_encoding[train_size:len(one_hot_encoding)]
    train_y, test_y = labels[0:train_size][:, None], labels[train_size:len(labels)][:, None]
    print("train_x: ", train_x.shape)
    print("train_y: ", train_y.shape)
    print("test_x: ", test_x.shape)
    print("test_y: ", test_y.shape)

    return train_x, train_y, test_x, test_y


import numpy as np

def cross_entropy_loss(y_true, y_pred):
    loss = - np.mean(y_true * np.log(y_pred + 1e-9))
    return loss

train_x, train_y, test_x, test_y = train_test_data()

mini_batch_size = 32
epochs = 1
lr = 0.001
model = ConvoModel(learning_rate=lr, batch_size =mini_batch_size)

for epoch in range(epochs):
    for i in tqdm(range(0, len(train_x), mini_batch_size)):
        x = train_x[i:i + mini_batch_size]
        y = train_y[i:i + mini_batch_size]
        # print("x: ", x.shape)
        # print("y: ", y.shape)
        output = model.forward(x)
        # print("output shape: ", output.shape)
        # print("output: ", output)
        # loss = cross_entropy_loss(output, y)
        # print("loss: ", loss)
        # delta = cross_entropy_loss(output, y, derivative=True)
        print("output: ", output.shape)
        print("y: ", y.shape)
        y_onehot = get_one_hot_encoding(y)
        print("y_onehot: ", y_onehot.shape)
        delta = output - y
        model.backward(delta=delta)

    print("Epoch: ", epoch)
    y_pred = model.forward(test_x)
    y_true = test_y

    # print("prediction value")

    # print(y_pred)
    # print(np.argmax(y_true, axis=1))
    # print("y_pred shape", np.argmax(y_pred, axis=1).shape)
    # print("y_true shape", y_true.shape)
    # print(np.argmax(y_pred, axis=1))
    # print()
    # print(y_true)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = y_pred[:, None]
    print("y_true shape: ", y_true.shape)
    print("y_pred shape: ", y_pred.shape)
    print("y_true: ", y_true)
    print("y_pred: ", y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    cross_entropy = cross_entropy_loss(y_pred, y_true)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    confusion_matrix_value = confusion_matrix(y_true, y_pred)
    print("confusion_matrix_value: ", confusion_matrix_value)
    print("macro-f1: ", macro_f1)
    print("cross entropy loss: ", cross_entropy)
    # validation_loss = validation_loss(y_pred, y_true)
    print("Accuracy: ", accuracy)



# loss vs epoch
# accuracy vs epoch
# f1 vs epoch
# confusion matrix
# loss vs learning rate for the last epoch





