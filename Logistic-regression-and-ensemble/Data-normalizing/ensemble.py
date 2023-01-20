import numpy as np
import copy

from data_handler import bagging_sampler


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        # create a list of base_estimator object
        # use deepcopy to create a new object
        self.base_estimator_list = []
        for i in range(self.n_estimator):
            self.base_estimator_list.append(copy.deepcopy(self.base_estimator))
        #self.base_estimator_list = [self.base_estimator for i in range(self.n_estimator)]


        # self.b = []
        # for i in range(self.n_estimator):
        #     self.b.append(0.1)
        #
        # self.w = []
        # for i in range(self.n_estimator):
        #     self.w.append(np.ones((shape, 1)) * 0.1)
        #
        for i in range(self.n_estimator):
            X_train, y_train = bagging_sampler(X, y)
            self.base_estimator_list[i].fit(X_train, y_train)

        # for i in range(self.n_estimator):
        #     print(self.base_estimator_list[i].w)
        #     print(self.base_estimator_list[i].b)

        return self


    def predict(self, X):
        y_pred = []
        for i in range(self.n_estimator):
            y_pred.append(self.base_estimator_list[i].predict(X))
        y_pred = np.array(y_pred)
        y_pred = np.sum(y_pred, axis=0)
        y_pred = np.where(y_pred > self.n_estimator/2, 1, 0)
        return y_pred
