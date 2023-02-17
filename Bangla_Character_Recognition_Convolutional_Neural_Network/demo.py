import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import tqdm
import pickle
import os

import matplotlib.pyplot as plt


def getWindows(input, output_size, filter_size, padding=0, stride=1, dilate=0):
    temp_input = input
    batch_size, channel_num, filter_h, filter_w = temp_input.strides

    if dilate != 0:
        temp_input = np.insert(temp_input, range(1, input.shape[2]), 0, axis=2)
        temp_input = np.insert(temp_input, range(1, input.shape[3]), 0, axis=3)

    if padding != 0:
        temp_input = np.pad(temp_input, pad_width=((0,), (0,), (padding,), (padding,)), mode='constant',
                            constant_values=(0.,))

    input_batch, input_channels, output_height, output_width = output_size
    output_batch, output_channels, _, _ = input.shape

    return np.lib.stride_tricks.as_strided(
        temp_input,
        (output_batch, output_channels, output_height, output_width, filter_size, filter_size),
        (batch_size, channel_num, stride * filter_h, stride * filter_w, filter_h, filter_w)
    )


class Convolution_Layer:
    def __init__(self, input_channels, output_channels, filter_size=3, stride=1, padding=0, learning_rate=0.01):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.saved_input = None

        self.learning_rate = learning_rate

        self.weight = 1e-3 * np.random.randn(self.output_channels, self.input_channels, self.filter_size,
                                             self.filter_size)
        self.bias = np.zeros(self.output_channels)

    def forward_prop(self, input):
        batches, channels, height, width = input.shape
        output_height = (height - self.filter_size + 2 * self.padding) // self.stride + 1
        output_width = (width - self.filter_size + 2 * self.padding) // self.stride + 1

        windows = getWindows(input, (batches, channels, output_height, output_width), self.filter_size, self.padding,
                             self.stride)

        output = np.einsum('bihwkl,oikl->bohw', windows, self.weight)

        output += self.bias[None, :, None, None]

        self.saved_input = input, windows
        return output

    def backward_prop(self, dout):
        input, windows = self.saved_input

        padding = self.filter_size - 1 if self.padding == 0 else self.padding

        dout_windows = getWindows(dout, input.shape, self.filter_size, padding=padding, stride=1,
                                  dilate=self.stride - 1)
        rot_kern = np.rot90(self.weight, 2, axes=(2, 3))

        grad_bias = np.sum(dout, axis=(0, 2, 3))
        grad_weight = np.einsum('bihwkl,bohw->oikl', windows, dout)
        grad_input = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)

        self.update_parameters(grad_bias, grad_weight)

        return grad_input

    def update_parameters(self, grad_bias, grad_weight):
        self.bias -= self.learning_rate * grad_bias
        self.weight -= self.learning_rate * grad_weight

    def print_weights(self):
        print("convolution layer: weight")
        print(self.weights)
        print("convolution layer: bias")
        print(self.biases)

    def cleanup(self):
        # self.input_channels = None
        # self.output_channels = None
        # self.filter_size = None
        # self.stride = None
        # self.padding = None

        self.saved_input = None

        # self.learning_rate = learning_rate


class ReLU_Activation_Layer:
    def forward_prop(self, input):
        self.input = input
        self.return_value = np.maximum(0, self.input)
        return self.return_value

    def backward_prop(self, grad_output):
        relu_grad = self.input > 0
        return grad_output * relu_grad

    def cleanup(self):
        self.input = None
        # self.return_value = None


class MaxPooling_Layer:

    def __init__(self, filter_dimension, stride=1):
        self.filter_dimension = filter_dimension
        self.stride = stride

    def forward_prop(self, input):

        self.input = input

        batch_size, num_channels, input_height, input_width = self.input.shape

        output_height = (input_height - self.filter_dimension) // self.stride + 1
        output_width = (input_width - self.filter_dimension) // self.stride + 1

        output = np.zeros((batch_size, num_channels, output_height, output_width))

        for x in range(output_width):
            for y in range(output_height):
                x_limit = x * self.stride + self.filter_dimension

                y_limit = y * self.stride + self.filter_dimension

                output[:, :, y, x] = np.max(self.input[:, :, y * self.stride: y_limit, x * self.stride: x_limit],
                                            axis=(2, 3))

        return output

    def backward_prop(self, grad_output):

        batch_size, num_channels, input_height, input_width = self.input.shape

        _, _, output_height, output_width = grad_output.shape

        grad_input = np.zeros(self.input.shape)

        for x in range(output_width):
            for y in range(output_height):
                x_limit = x * self.stride + self.filter_dimension
                y_limit = y * self.stride + self.filter_dimension

                input_region = self.input[:, :, y * self.stride: y_limit, x * self.stride: x_limit]

                max_value = np.max(input_region, axis=(2, 3), keepdims=True)
                max_mask = (input_region == max_value)

                grad_input[:, :, y * self.stride: y_limit, x * self.stride: x_limit] += max_mask * grad_output[:, :, y,
                                                                                                   x][:, :, None, None]

        return grad_input

    def cleanup(self):
        self.input = None
        # self.filter_dimension = None
        # self.stride = None


class Flattening_Layer:
    def forward_prop(self, input):
        self.input = input
        self.input_shape = self.input.shape
        return input.reshape(self.input_shape[0], -1)

    def backward_prop(self, grad_output):
        return grad_output.reshape(self.input_shape)

    def cleanup(self):
        self.input = None
        # self.input_shape = None


class Fully_Connected_Layer:
    def __init__(self, output_dimension, learning_rate=0.01):
        self.weights = None
        self.biases = None
        self.output_dimension = output_dimension

        self.learning_rate = learning_rate

    def forward_prop(self, input):

        self.input = input

        self.input_dimension = input.shape[1]

        if self.weights is None:
            self.weights = np.random.randn(self.input_dimension, self.output_dimension) * np.sqrt(
                2.0 / (self.input_dimension + self.output_dimension))

        if self.biases is None:
            self.biases = np.zeros((1, self.output_dimension))

        return self.input.dot(self.weights) + self.biases

    def backward_prop(self, grad_output):
        grad_input = grad_output.dot(self.weights.T)
        grad_weights = self.input.T.dot(grad_output)

        grad_biases = np.sum(grad_output, axis=0, keepdims=True)

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        # self.weights = np.clip(self.weights, -1, 1)
        # self.biases = np.clip(self.biases, -1, 1)

        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input

    def print_weights(self):
        print("Fully connected layer: weight")
        print(self.weights)
        print("Fully connected layer: bias")
        print(self.biases)

    def cleanup(self):
        self.input = None
        # self.output_dimension = None
        # self.learning_rate = None


class SoftMax_Layer:
    def forward_prop(self, input):
        self.input = input

        softmax_exp = np.exp(self.input - np.max(self.input, axis=1, keepdims=True))
        self.softmax_prob = softmax_exp / np.sum(softmax_exp, axis=1, keepdims=True)
        return self.softmax_prob

    def backward_prop(self, input):
        return self.input

    def cleanup(self):
        self.input = None
        self.softmax_prob = None


def calc_CrossEntropy(y_predict, y_true):
    y_predict = np.clip(y_predict, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_predict), axis=-1))


def calc_Accuracy(softmax_values, validation_labels):
    return np.mean(np.argmax(softmax_values, axis=1) == np.argmax(validation_labels, axis=1))


def shuffle_data(data, labels, test_ratio):
    combined = np.column_stack((data, labels))

    np.random.shuffle(combined)

    data = combined[:, :-1]
    labels = combined[:, -1]

    n_test = int(test_ratio * data.shape[0])

    data_test = data[:n_test]
    labels_test = labels[:n_test, ]

    data_train = data[n_test:]
    labels_train = labels[n_test:]

    data_train = data_train.reshape(data_train.shape[0])
    data_test = data_test.reshape(data_test.shape[0])
    labels_train = labels_train.reshape(labels_train.shape[0])
    labels_test = labels_test.reshape(labels_test.shape[0])

    labels_train = np.array(labels_train, dtype=np.int32)
    labels_test = np.array(labels_test, dtype=np.int32)

    return data_train, labels_train, data_test, labels_test


def process_images(images_list, img_path_dir):
    all_images = []

    for i in range(len(images_list)):
        temp_dir = img_path_dir
        if images_list[i].startswith("a"):
            temp_dir += "training-a/"
        elif images_list[i].startswith("b"):
            temp_dir += "training-b/"
        elif images_list[i].startswith("c"):
            temp_dir += "training-c/"

        img_path = os.path.join(temp_dir, images_list[i])

        img = cv2.imread(img_path)

        img = cv2.resize(img, (28, 28))

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray_img = gray_img.astype('float32') / 255.

        gray_img = gray_img.reshape(1, gray_img.shape[0], gray_img.shape[1])

        all_images.append(gray_img)

    all_images = np.array(all_images)

    mean_image = np.mean(all_images)
    standard_deviation = np.std(all_images)
    all_images = (all_images - mean_image) / standard_deviation

    return all_images


def load_Train_Dataset(img_path_dir, csv_dir_list):
    csv_df1 = pd.read_csv(csv_dir_list[0])
    csv_df2 = pd.read_csv(csv_dir_list[1])
    csv_df3 = pd.read_csv(csv_dir_list[2])

    # print(csv_df1.shape[0], csv_df2.shape[0], csv_df3.shape[0])

    csv_df_merged = pd.concat([csv_df1, csv_df2])
    csv_df_merged = pd.concat([csv_df_merged, csv_df3.sample(n=csv_df3.shape[0] // 5)])

    # print(csv_df_merged.shape[0])

    all_filename = csv_df_merged["filename"].values
    all_labels = csv_df_merged["digit"].values

    train_data, train_labels, test_data, test_labels = shuffle_data(all_filename, all_labels, 0.2)

    train_data = process_images(train_data, img_path_dir)
    test_data = process_images(test_data, img_path_dir)

    return train_data, train_labels, test_data, test_labels


def split_train_test_data():
    # csv_dir = './dataset/NumtaDB_with_aug/training-b.csv'
    img_path_dir = './dataset/NumtaDB_with_aug/'

    csv_dir_list = ['./dataset/NumtaDB_with_aug/training-a.csv', './dataset/NumtaDB_with_aug/training-b.csv',
                    './dataset/NumtaDB_with_aug/training-c.csv']
    # img_path_dir_list = ['./dataset/NumtaDB_with_aug/training-a/', './dataset/NumtaDB_with_aug/training-b/', './dataset/NumtaDB_with_aug/training-c/']

    train_data, train_labels, test_data, test_labels = load_Train_Dataset(img_path_dir, csv_dir_list)

    train_labels = make_one_hot(train_labels)
    test_labels = make_one_hot(test_labels)

    return train_data, train_labels, test_data, test_labels


def make_one_hot(labels, n_unique_labels=10):
    n_labels = len(labels)
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def one_hot_to_digit(one_hot):
    return np.argmax(one_hot, axis=1)


class CNN_Model():

    def __init__(self, learning_rate):
        self.allLayers = []

        self.convObj1 = Convolution_Layer(1, 6, 5, 1, 1, learning_rate)
        self.reluObj1 = ReLU_Activation_Layer()
        self.maxPoolObj1 = MaxPooling_Layer(2, 2)

        self.convObj2 = Convolution_Layer(6, 16, 5, 1, 1, learning_rate)
        self.reluObj2 = ReLU_Activation_Layer()
        self.maxPoolObj2 = MaxPooling_Layer(2, 2)

        self.flatteningObj1 = Flattening_Layer()

        self.fullyConnectedObj1 = Fully_Connected_Layer(120, learning_rate)
        self.reluObj3 = ReLU_Activation_Layer()

        self.fullyConnectedObj2 = Fully_Connected_Layer(84, learning_rate)
        self.reluObj4 = ReLU_Activation_Layer()

        self.fullyConnectedObj3 = Fully_Connected_Layer(10, learning_rate)
        self.softmaxObj1 = SoftMax_Layer()

    def forward_prop(self, input):
        intermediate_value = self.convObj1.forward_prop(input)
        intermediate_value = self.reluObj1.forward_prop(intermediate_value)
        intermediate_value = self.maxPoolObj1.forward_prop(intermediate_value)

        intermediate_value = self.convObj2.forward_prop(intermediate_value)
        intermediate_value = self.reluObj2.forward_prop(intermediate_value)
        intermediate_value = self.maxPoolObj2.forward_prop(intermediate_value)

        intermediate_value = self.flatteningObj1.forward_prop(intermediate_value)

        intermediate_value = self.fullyConnectedObj1.forward_prop(intermediate_value)
        intermediate_value = self.reluObj3.forward_prop(intermediate_value)

        intermediate_value = self.fullyConnectedObj2.forward_prop(intermediate_value)
        intermediate_value = self.reluObj4.forward_prop(intermediate_value)

        intermediate_value = self.fullyConnectedObj3.forward_prop(intermediate_value)
        intermediate_value = self.softmaxObj1.forward_prop(intermediate_value)

        return intermediate_value

    def backward_prop(self, input):
        intermediate_value = self.softmaxObj1.backward_prop(input)
        intermediate_value = self.fullyConnectedObj3.backward_prop(intermediate_value)

        intermediate_value = self.reluObj4.backward_prop(intermediate_value)
        intermediate_value = self.fullyConnectedObj2.backward_prop(intermediate_value)

        intermediate_value = self.reluObj3.backward_prop(intermediate_value)
        intermediate_value = self.fullyConnectedObj1.backward_prop(intermediate_value)

        intermediate_value = self.flatteningObj1.backward_prop(intermediate_value)

        intermediate_value = self.maxPoolObj2.backward_prop(intermediate_value)
        intermediate_value = self.reluObj2.backward_prop(intermediate_value)
        intermediate_value = self.convObj2.backward_prop(intermediate_value)

        intermediate_value = self.maxPoolObj1.backward_prop(intermediate_value)
        intermediate_value = self.reluObj1.backward_prop(intermediate_value)
        intermediate_value = self.convObj1.backward_prop(intermediate_value)

        return intermediate_value

    def validation_test(self, validation_data, validation_labels):
        softmax_values = self.forward_prop(validation_data)

        loss = calc_CrossEntropy(softmax_values, validation_labels)

        accuracy = calc_Accuracy(softmax_values, validation_labels)

        modified_softmax_values = np.argmax(softmax_values, axis=1)
        modified_validation_labels = np.argmax(validation_labels, axis=1)

        f1_score_val = f1_score(modified_validation_labels, modified_softmax_values, average='macro')

        confusion_matrix_value = confusion_matrix(modified_validation_labels, modified_softmax_values)

        return accuracy, loss, f1_score_val, softmax_values, confusion_matrix_value

    def calTrainingLoss(self, train_data, train_labels):
        softmax_values = self.forward_prop(train_data)

        loss = calc_CrossEntropy(softmax_values, train_labels)

        return loss

    def cleanup(self):
        self.convObj1.cleanup()
        self.reluObj1.cleanup()
        self.maxPoolObj1.cleanup()

        self.flatteningObj1.cleanup()
        self.fullyConnectedObj1.cleanup()
        self.reluObj2.cleanup()
        self.fullyConnectedObj2.cleanup()
        self.softmaxObj1.cleanup()

    def train_model():
        pass

    def predict(self, img):
        img = cv2.resize(img, (28, 28))

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray_img = gray_img.astype('float32') / 255.

        gray_img = gray_img.reshape(1, gray_img.shape[0], gray_img.shape[1])

        input_img = np.array(gray_img)

        input = np.expand_dims(input_img, axis=0)

        intermediate_vlue = self.forward_prop(input)
        return intermediate_vlue


def train_one_epoch(model, X, y, validation_data, validation_labels, batch_size=32):
    num_samples = X.shape[0]

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    num_batches = num_samples // batch_size
    batches = np.array_split(indices, num_batches)

    for batch_indices in batches:
        # Select the mini-batch data
        batch_data = X[batch_indices]
        batch_labels = y[batch_indices]

        # forward prop
        outputs = model.forward_prop(batch_data)

        # backward prop
        grads = model.backward_prop(outputs - batch_labels)

    validation_accuracy, validation_loss, f1_score, softmax_values, confusion_matrix_value = model.validation_test(
        validation_data, validation_labels)

    training_loss = model.calTrainingLoss(X, y)

    return validation_accuracy, validation_loss, f1_score, softmax_values, confusion_matrix_value, training_loss


def main():
    num_of_epochs = 10
    learning_rate = float(input("Enter learning rate: "))

    model = CNN_Model(learning_rate)
    train_data, train_labels, test_data, test_labels = split_train_test_data()

    val_acc_list = []
    val_loss_list = []
    f1_score_list = []
    training_loss_list = []

    for i in range(num_of_epochs):
        val_accuracy, val_loss, f1_score, softmax_values, confusion_matrix_value, training_loss = train_one_epoch(model,
                                                                                                                  train_data,
                                                                                                                  train_labels,
                                                                                                                  test_data,
                                                                                                                  test_labels,
                                                                                                                  10)
        print("iteration: {} accuracy: {} f1_score: {} loss: {} training_loss {}".format(i, val_accuracy, f1_score,
                                                                                         val_loss, training_loss))

        print("confusion_matrix: ", confusion_matrix_value)

        # print(one_hot_to_digit(softmax_values))
        # print("softmax_values: ", softmax_values)
        val_acc_list.append(val_accuracy)
        val_loss_list.append(val_loss)
        f1_score_list.append(f1_score)
        training_loss_list.append(training_loss)

    # DPI = 300

    plt.plot(range(num_of_epochs), val_acc_list, label="Validation Accuracy")
    plt.title("Validation accuracy for learning rate = {}".format(learning_rate))
    plt.xlabel("epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("val_accuracy.png")

    plt.show()
    # plt.savefig("val_accuracy.png", dpi=DPI)

    plt.plot(range(num_of_epochs), val_loss_list, label="Validation Loss")
    plt.title("Validation Loss for learning rate = {}".format(learning_rate))
    plt.xlabel("epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig("val_loss.png")
    plt.show()

    plt.plot(range(num_of_epochs), training_loss_list, label="Training Loss")
    plt.title("Training Loss for learning rate = {}".format(learning_rate))
    plt.xlabel("epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.show()

    plt.plot(range(num_of_epochs), f1_score_list, label="Macro F1 Score")
    plt.title("Macro F1 score for learning rate = {}".format(learning_rate))
    plt.xlabel("epoch")
    plt.ylabel("Macro F1 score")
    plt.legend()
    plt.savefig("macro_f1.png")
    plt.show()

    model.cleanup()
    with open('1705018_model.pickle', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()

