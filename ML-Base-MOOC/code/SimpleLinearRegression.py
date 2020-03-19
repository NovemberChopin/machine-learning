import numpy as np
from metrics import r2_score

class SimpleLinearRegression:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, "Simple Linear Regression can only solve single feature train data"
        assert len(x_train) == len(y_train), \
            "The size of x_train must be equal to the size of y_train"
        
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, \
            "Simple Linear Regression can only solve single feature train data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict"

        return np.array([ self._predict(i) for i in x_predict])

    def _predict(self, x_predict):
        return self.a_ * x_predict + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)