import pandas as pd
import numpy as np


def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement
    data = pd.read_csv('parkinsons.csv')
    #X = data.iloc[:, :-1].values
    #y = data.iloc[:, -1].values
    y = data['status'].values
    X = data.drop(['status', 'name'], axis=1).values

    # normalize each column
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i] - np.min(X[:, i])))
        #X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
    print(X.shape, y.shape)
    print('X', X)
    print('y', y)


    return X, y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    #add y to X
    dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    #shuffle
    if shuffle:
        np.random.shuffle(dataset)
    #split

    train_size = int(len(dataset)*(1-test_size))
    X_train = dataset[:train_size, :-1]

    # convert numpy array values to int

    y_train = dataset[:train_size, -1].astype(int)
    X_test = dataset[train_size:, :-1]
    y_test = dataset[train_size:, -1].astype(int)

    #reshape y_train and y_test as a (n,1) array
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return X_train, y_train, X_test, y_test



def bagging_sampler(X, y):

    #Randomly sample with replacement
    #Size of sample will be same as input data
    #Return X_sample, y_sample

    n =X.shape[0]
    sample = np.random.choice(n, n, replace=True)
    #print('sample', sample)
    X_sample = X[sample]
    y_sample = y[sample]
    return X_sample, y_sample

