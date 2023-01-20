import numpy as np

class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.learning_rate = params['learning_rate']
        self.iterations = params['iterations']
        self.m = params['m']
        self.n = params['n']


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

        self.b = 1
        self.w = np.ones((self.n, 1)) * 0.1

        for i in range(self.iterations):
            z = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(z)
            dw = (1 / self.m) * np.dot(X.T, (y_pred - y))
            db = (1 / self.m) * np.sum(y_pred - y)
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

        return self


    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y_pred = self.sigmoid(z)
        y = np.where(y_pred > 0.5, 1, 0)
        return y



