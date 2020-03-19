import numpy as np
from sklearn import datasets


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X_train):
        # 根据训练数据集 X 获得数据的均值和方差
        assert X_train.ndim == 2, "The dimension of X must be 2"

        self.mean_ = np.array([np.mean(X_train[:, i]) for i in range(X_train.shape[1])])
        self.scale_ = np.array([np.std(X_train[:, i]) for i in range(X_train.shape[1])])
        # print("self.mean: ", self.mean_)
        # print("self.scale: ", self.scale_)
        return self

    def transform(self, X):
        # 将 X 根据这个StandardScaler进行均值方差归一化处理
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
            "must fit before transform"
        assert X.shape[1] == len(self.mean_), \
            "the feature number of X must be equal to mean_ and scale_"

        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return  resX

def main():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X = X[y < 50.0]
    y = y[y < 50.0]
    std = StandardScaler()
    print(X.shape)
    std.fit(X)
    X_std = std.transform(X)
    print(X_std.shape)

if __name__ == "__main__":
    main()