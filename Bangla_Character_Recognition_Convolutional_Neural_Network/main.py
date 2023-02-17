import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from ConvolutionLayer import Convolution
import cv2


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
    pass



# train = pd.read_csv(csv_path)
# print(train)
# labels = {}
#
# image_name = train['filename']
# label = train['digit']
#
