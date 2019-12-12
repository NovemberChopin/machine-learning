import numpy as np
from math import sqrt
from sklearn import datasets
from collections import Counter

from .model_selection import train_test_split

class KNNClassifier():
    def __init__(self, k):
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"

        return [self._predict(x_predict) for x_predict in X_predict]

    def _predict(self, x):
        """给定单个待预测的数据，返回x的预测结果值"""
        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return np.sum(y_test == y_predict) / len(X_test)

def main():
    # raw_data_X = [[3.4, 2.3],
    #               [3.1, 1.8],
    #               [1.3, 3.4],
    #               [3.6, 4.7],
    #               [2.3, 2.9],
    #               [7.4, 4.7],
    #               [5.7, 3.5],
    #               [9.2, 2.5],
    #               [7.8, 3.4],
    #               [7.9, 0.8],
    #               ]
    # raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # X_train = np.array(raw_data_X)
    # y_train = np.array(raw_data_y)
    #
    # my_kNN_clf = KNNClassifier(k=3)
    # my_kNN_clf.fit(X_train, y_train)
    # a = my_kNN_clf.predict(np.array([8.1, 7.4]).reshape(1, -1))
    # print(a)

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, y_train, X_test, y_test = train_test_split(X, y, seed=123)

    my_knn_clf = KNNClassifier(k=3)
    my_knn_clf.fit(X_train, y_train)
    my_knn_clf.score(X_train, X_test)

if __name__ == "__main__":
    main()