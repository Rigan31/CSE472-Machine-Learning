import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import cv2
import math
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix



class ConvolutionLayer:
    def __init__(self, input_channel, output_channel, filters_dim, stride, padding):
        self.input_shape = input_channel
        self.num_filters = output_channel
        self.filter_shape = filters_dim
        self.stride = stride
        self.padding = padding
        self.learning_rate = 0.001

        self.weights = None
        self.bias = None
        self.input_pad = None

    def forward(self, u):
        num_samples = u.shape[0]
        input_dim = u.shape[1]
        output_dim = math.floor((input_dim - self.filter_shape + 2 * self.padding) / self.stride) + 1
        num_channels = u.shape[3]

        if self.weights is None:
            self.weights = np.random.randn(self.num_filters, self.filter_shape, self.filter_shape,num_channels) * math.sqrt(2 / (self.filter_shape * self.filter_shape * num_channels))
        if self.bias is None:
            self.bias = np.zeros(self.num_filters)

        self.input_pad = np.pad(u, ((0,), (self.padding,), (self.padding,), (0,)), mode='constant')
        v = np.zeros((num_samples, output_dim, output_dim, self.num_filters))

        for k in range(num_samples):
            for l in range(self.num_filters):
                for i in range(output_dim):
                    for j in range(output_dim):
                        v[k, i, j, l] = np.sum(self.input_pad[k, i * self.stride: i * self.stride + self.filter_shape,
                                               j * self.stride: j * self.stride + self.filter_shape, :] * self.weights[
                                                   l]) + self.bias[l]

        return v

    # def backward(self, del_v):
    #     num_samples = del_v.shape[0]
    #     input_dim = del_v.shape[1]
    #     input_dim_pad = (input_dim - 1) * self.stride + 1
    #     output_dim = self.input_pad.shape[1] - 2 * self.padding
    #     num_channels = self.input_pad.shape[3]
    #
    #     del_u = np.zeros((num_samples, output_dim, output_dim, num_channels))
    #     del_weights = np.zeros((self.num_filters, self.filter_shape, self.filter_shape, num_channels))
    #     del_bias = np.zeros(self.num_filters)
    #
    #     for k in range(num_samples):
    #         for l in range(self.num_filters):
    #             for i in range(input_dim):
    #                 for j in range(input_dim):
    #                     del_u[k, i * self.stride: i * self.stride + self.filter_shape, j * self.stride: j * self.stride + self.filter_shape, :] += self.weights[l] * del_v[k, i, j, l]
    #                     del_weights[l] += self.input_pad[k, i * self.stride: i * self.stride + self.filter_shape, j * self.stride: j * self.stride + self.filter_shape, :] * del_v[k, i, j, l]
    #                     del_bias[l] += del_v[k, i, j, l]
    #
    #     self.weights -= self.learning_rate * del_weights
    #     self.bias -= self.learning_rate * del_bias
    #
    #     return del_u[:, self.padding: self.padding + input_dim_pad, self.padding: self.padding + input_dim_pad, :]
    #
    def backward(self, del_v):
        num_samples = del_v.shape[0]
        input_dim = del_v.shape[1]
        input_dim_pad = (input_dim - 1) * self.stride + 1
        output_dim = self.input_pad.shape[1] - 2 * self.padding
        num_channels = self.input_pad.shape[3]

        del_b = np.sum(del_v, axis=(0, 1, 2)) / num_samples
        # print("del_b value: ", del_b[0])
        # print("del_v value: ", del_v)
        del_v_sparse = np.zeros((num_samples, input_dim_pad, input_dim_pad, self.num_filters))
        del_v_sparse[:, :: self.stride, :: self.stride, :] = del_v
        weights_prime = np.rot90(np.transpose(self.weights, (3, 1, 2, 0)), 2, axes=(1, 2))
        del_w = np.zeros((self.num_filters, self.filter_shape, self.filter_shape, num_channels))
        # print("del_w value: ", del_w[0, 0, 0, 0])

        for l in range(self.num_filters):
            for i in range(self.filter_shape):
                for j in range(self.filter_shape):
                    del_w[l, i, j, :] = np.mean(np.sum(
                        self.input_pad[:, i: i + input_dim_pad, j: j + input_dim_pad, :] * np.reshape(
                            del_v_sparse[:, :, :, l], del_v_sparse.shape[: 3] + (1,)), axis=(1, 2)), axis=0)

        del_u = np.zeros((num_samples, output_dim, output_dim, num_channels))
        del_v_sparse_pad = np.pad(del_v_sparse, (
        (0,), (self.filter_shape - 1 - self.padding,), (self.filter_shape - 1 - self.padding,), (0,)), mode='constant')

        for k in range(num_samples):
            for l in range(num_channels):
                for i in range(output_dim):
                    for j in range(output_dim):
                        del_u[k, i, j, l] = np.sum(
                            del_v_sparse_pad[k, i: i + self.filter_shape, j: j + self.filter_shape, :] * weights_prime[
                                l])

        self.weights -= self.learning_rate * del_w
        self.bias -= self.learning_rate * del_b
        # self.update_learnable_parameters(del_w, del_b, lr)
        # print("weights convo layer: ", self.weights[0, 0, 0, 0])
        # print("bias convo layer: ", self.bias[0])
        # print("del_w convo layer: ", del_w)
        # print("del_b convo layer: ", del_b)
        # print("bias convo layer: ", self.bias[0, 0])
        # print("dw: ", del_w[0, 0, 0, 0])
        # print("db: ", del_b[0])
        return del_u


class ReLUActivation:
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, delta):
        return delta * (self.input > 0)


class MaxPoolingLayer:
    def __init__(self, filter_shape, stride):
        self.filter_shape = filter_shape
        self.stride = stride
        self.input_shape = None
        self.save_output = None

    def forward(self, input):
        self.input_shape = input.shape
        sample_size, input_height, input_width, input_channel = input.shape
        output_height = int((input_height - self.filter_shape) / self.stride) + 1
        output_width = int((input_width - self.filter_shape) / self.stride) + 1
        output = np.zeros((sample_size, output_height, output_width, input_channel))
        self.save_output = np.zeros((sample_size, output_height, output_width, input_channel)).astype(np.int32)

        for i in range(sample_size):
            for h in range(output_height):
                for w in range(output_width):
                    for c in range(input_channel):
                        output[i, h, w, c] = np.max(input[i, h * self.stride: h * self.stride + self.filter_shape,
                                                    w * self.stride: w * self.stride + self.filter_shape, c])
                        self.save_output[i, h, w, c] = np.argmax(
                            input[i, h * self.stride: h * self.stride + self.filter_shape,
                            w * self.stride: w * self.stride + self.filter_shape, c])

        return output

    def backward(self, del_v):
        del_u = np.zeros(self.input_shape)

        num_samples = del_v.shape[0]
        input_dim = del_v.shape[1]
        num_channels = del_v.shape[3]

        for k in range(num_samples):
            for l in range(num_channels):
                for i in range(input_dim):
                    for j in range(input_dim):
                        position = tuple(sum(pos) for pos in zip((self.save_output[k, i, j, l] // self.filter_shape,
                                                                  self.save_output[k, i, j, l] % self.filter_shape),
                                                                 (i * self.stride, j * self.stride)))
                        del_u[(k,) + position + (l,)] = del_u[(k,) + position + (l,)] + del_v[k, i, j, l]

        return del_u
    # eta amar code chilo
    # def backward(self, delta_output):
    #     delta_input = np.zeros(self.input_shape)
    #
    #     sample_size, output_height, output_width, output_channel = delta_output.shape
    #     # input_height = int((self.input_shape[1] - self.filter_shape) / self.stride) + 1
    #     # input_width = int((self.input_shape[2] - self.filter_shape) / self.stride) + 1
    #
    #     for i in range(sample_size):
    #         for h in range(output_height):
    #             for w in range(output_width):
    #                 for c in range(output_channel):
    #                     h_start = h * self.stride
    #                     # h_end = h * self.stride + self.filter_shape
    #                     w_start = w * self.stride
    #                     # w_end = w * self.stride + self.filter_shape
    #
    #                     max_index = np.unravel_index(self.save_output[i, h, w, c], (self.filter_shape, self.filter_shape))
    #                     delta_input[i, h_start + max_index[0], w_start + max_index[1], c] = delta_output[i, h, w, c]
    #
    #     return delta_input

    # def backward(self, delta):
    #     delta = delta.reshape(self.input_shape)
    #     sample_size, input_height, input_width, input_channel = delta.shape
    #     output_height = int((input_height - self.filter_shape) / self.stride) + 1
    #     output_width = int((input_width - self.filter_shape) / self.stride) + 1
    #     output = np.zeros((sample_size, output_height, output_width, input_channel))
    #
    #     for i in range(sample_size):
    #         for h in range(output_height):
    #             for w in range(output_width):
    #                 for c in range(input_channel):
    #                     output[i, h, w, c] = np.max(delta[i, h * self.stride: h * self.stride + self.filter_shape, w * self.stride: w * self.stride + self.filter_shape, c])
    #
    #     return output

    # def backward(self, delta):
    #     sample_size, input_height, input_width, input_channel = self.input_shape
    #     output_height = int((input_height - self.filter_shape) / self.stride) + 1
    #     output_width = int((input_width - self.filter_shape) / self.stride) + 1
    #     grad_input = np.zeros(self.input_shape)
    #
    #     for k in range(sample_size):
    #         for l in range(input_channel):
    #             for i in range(input_width):
    #                 for j in range(input_height):
    #                     position = tuple(sum(pos) for pos in zip((self.save_output[k, i, j, l] // self.filter_shape,
    #                                                               self.save_output[k, i, j, l] % self.filter_shape),
    #                                                              (i * self.stride, j * self.stride)))
    #                     grad_input[(k,) + position + (l,)] = grad_input[(k,) + position + (l,)] + delta[k, i, j, l]
    #
    #     return grad_input

    # def backward(self, delta):
    #     sample_size, input_height, input_width, input_channel = self.input_shape
    #     output_height = int((input_height - self.filter_shape) / self.stride) + 1
    #     output_width = int((input_width - self.filter_shape) / self.stride) + 1
    #     grad_input = np.zeros(self.input_shape)
    #
    #     for k in range(sample_size):
    #         for l in range(input_channel):
    #             for i in range(input_width):
    #                 for j in range(input_height):
    #                     position = tuple(sum(pos) for pos in zip((self.v_map[k, i, j, l] // self.filter_shape,
    #                                                               self.v_map[k, i, j, l] % self.filter_shape),
    #                                                              (i * self.stride, j * self.stride)))
    #                     grad_input[(k,) + position + (l,)] = grad_input[(k,) + position + (l,)] + grad_input[k, i, j, l]
    #
    #     return grad_input


class FlattenLayer:
    def __init__(self):
        self.input_shape = None

    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, delta):
        return delta.reshape(self.input_shape)


class FullyConnectedLayer:
    def __init__(self, output_size, learning_rate=0.001):
        self.output_size = output_size
        self.weights = None
        self.bias = None
        self.dw = None
        self.db = None
        self.input = None
        self.learning_rate = learning_rate

    def forward(self, input):
        self.input = input
        input_dim = input.shape[1]
        if self.weights is None:
            self.weights = np.random.randn(input_dim, self.output_size) / np.sqrt(input_dim / 2)
            self.bias = np.zeros((1, self.output_size))
            # self.bias = np.zeros((self.output_size, 1))
        # print("input shape: ", input.shape)
        # print("weights shape: ", self.weights.shape)
        # print("bias shape: ", self.bias.shape)
        return np.dot(input, self.weights) + self.bias

    def backward(self, delta):
        delta = np.array(delta)
        self.dw = np.dot(self.input.T, delta) / delta.shape[1]
        # self.db = np.sum(delta, axis=0).reshape(1, -1)
        self.db = np.mean(delta, axis=0, keepdims=True)
        # print("delta shape: ", delta.shape)
        # print("input shape: ", np.transpose(self.input).shape)
        # print("delta shape: ", delta.shape)
        # self.dw = (np.transpose(self.input) @delta ) / delta.shape[1]
        # self.db = np.reshape(np.mean(delta, axis=1), (delta.shape[0], 1))
        self.weights -= self.learning_rate * self.dw
        self.bias -= self.learning_rate * self.db
        # print(type(self.db))
        # print("dw shape: ", self.dw.shape)
        # print("db shape: ", self.db.shape)
        # print("self.bias fully connected layer: ", self.bias[0, 0])
        # print("self.weights fully connected layer: ", self.weights[0, 0])
        # print("self.dw fully connected layer: ", self.dw[0, 0])
        # print("self.db fully connected layer: ", self.db[0, 0])
        return np.dot(delta, self.weights.T)


class SoftMaxLayer:
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        # exp = np.exp(input)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, delta):
        # print("type in softmax layer: ", type(delta))
        return delta

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SigmoidLayer:
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return sigmoid(input)

    def backward(self, delta):
        return delta * sigmoid(self.input) * (1 - sigmoid(self.input))

class ConvoModel:

    def __init__(self, learning_rate=0.001, batch_size=32):
        self.input_channels = 1
        self.output_channels = 6
        self.filter_dim = 5
        self.stride = 1
        self.padding = 1
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.convolutionLayer = ConvolutionLayer(self.input_channels, self.output_channels, self.filter_dim,
                                                 self.stride, self.padding)
        self.reluLayer = ReLUActivation()
        self.maxPoolingLayer = MaxPoolingLayer(filter_shape=2, stride=2)
        self.flattenLayer = FlattenLayer()
        self.fullyConnectedLayer1 = FullyConnectedLayer(output_size=10, learning_rate=0.001)
        self.sigmoidLayer = SigmoidLayer()
        # self.fullyConnectedLayer2 = FullyConnectedLayer(output_size=10, learning_rate=0.001)
        # self.fullyConnectedLayer3 = FullyConnectedLayer(output_size=10, learning_rate=0.0001)

        self.softMaxLayer = SoftMaxLayer()

    def forward(self, x):
        x = self.convolutionLayer.forward(x)
        # print("ConvoModel forward: ", x.shape)
        x = self.reluLayer.forward(x)
        # print("ReLU Layer: ", x.shape)
        x = self.maxPoolingLayer.forward(x)
        # print("MaxPooling Layer: ", x.shape)
        x = self.flattenLayer.forward(x)
        # print("Flatten Layer: ", x.shape)
        x = self.fullyConnectedLayer1.forward(x)
        x = self.sigmoidLayer.forward(x)
        # x = self.fullyConnectedLayer2.forward(x)
        # x = self.fullyConnectedLayer3.forward(x)

        # print("FullyConnected Layer: ", x.shape)
        x = self.softMaxLayer.forward(x)
        # print("SoftMax Layer: ", x.shape)
        print("\n")
        # print("x: ", x)
        # print("ConvoModel forward: ", x.shape)
        return x

    def backward(self, delta):
        # print("delta: ", delta)
        delta = self.softMaxLayer.backward(delta)
        # print("Backward SoftMax Layer: ", delta.shape)
        # delta = self.fullyConnectedLayer3.backward(delta)

        # delta = self.fullyConnectedLayer2.backward(delta)
        delta = self.sigmoidLayer.backward(delta)
        # delta = sigmoid(delta)
        delta = self.fullyConnectedLayer1.backward(delta)
        # delta = sigmoid(delta)
        # print("Backward FullyConnected Layer: ", delta.shape)
        delta = self.flattenLayer.backward(delta)
        # print("Backward Flatten Layer: ", delta.shape)
        delta = self.maxPoolingLayer.backward(delta)
        # print("Backward MaxPooling Layer: ", delta.shape)
        delta = self.reluLayer.backward(delta)
        # print("Backward ReLU Layer: ", delta.shape)
        delta = self.convolutionLayer.backward(delta)
        # print("Backward Convo Layer: ", delta.shape)
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
    image_path = './dataset/NumtaDB_with_aug/training-a//'
    csv_path = './dataset/NumtaDB_with_aug/training-a.csv'

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
        delta = output - y
        model.backward(delta=delta)

    print("Epoch: ", epoch)
    y_pred = model.forward(test_x)
    y_true = test_y

    print("y_pred: ", y_pred)

    # print("prediction value")

    # print(y_pred)
    # print(np.argmax(y_true, axis=1))
    # print("y_pred shape", np.argmax(y_pred, axis=1).shape)
    # print("y_true shape", y_true.shape)
    # print(np.argmax(y_pred, axis=1))
    # print()
    # print(y_true)
    y_pred = np.argmax(y_pred, axis=1)
    print("y_pred 2", y_pred)
    y_pred = y_pred[:, None]
    # print("y_true shape: ", y_true.shape)
    # print("y_pred shape: ", y_pred.shape)
    # print("y_true: ", y_true)
    # print("y_pred: ", y_pred)
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





