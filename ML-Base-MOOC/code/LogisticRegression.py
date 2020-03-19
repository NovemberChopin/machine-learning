import  numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

from metrics import accuracy_score
from model_slection import train_test_splice
from preprocessing import StandardScaler

class LogisticRegression:
    """docstring for LogisticRegression"""
    def __init__(self):
        self.coef_ = None   # theta[1-n]
        self.intercept_ = None  # theta0
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y): # 损失函数
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return - np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(X_b)
            except:
                return float('inf')

        def DJ(theta, X_b, y):  # 求导
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y)  / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = DJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1
            return theta

        X_b = np.hstack([np.ones([len(X_train), 1]), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return  self

    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        """根据训练数据集X_train, y_train, 使用随机梯度下降法训练Linear Regression模型"""
        # n_iters 代表对所有样本计算几遍
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1

        # # 此时传入的是某一行的数据
        def dj_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta, n_iters, t0=5, t1=50):

            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)
            # iteration n_iters times for all data
            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dj_sgd(theta, X_b_new[i], y_new[i])
                    # 此时 cur_iter 代表循环遍数，经过处理后传入 learning_rate 函数中
                    theta = theta - learning_rate(n_iters * m + i) * gradient
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.rand(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict_proba(self, X_predict):
        """给定待预测数据集 X_predict，返回表示 X_predict 的概率结果 y_hat"""

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量 y_hat"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

def plot(X, y):
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
    plt.show()

def main():
    boston = datasets.load_iris()
    X = boston.data
    y = boston.target
    X = X[y<2, :2]
    y = y[y<2]
    plot(X, y)
    X_train, y_train, X_test, y_test = train_test_splice(X, y, seed=666)

    reg = LogisticRegression()

    std = StandardScaler()
    std.fit(X_train)
    X_train_std = std.transform(X_train)
    X_test_std = std.transform(X_test)

    # fit
    reg.fit(X_train_std, y_train)
    score = reg.score(X_test_std, y_test)


    print("intercept_:", reg.intercept_)
    print("coef_:", reg.coef_)
    print("score: ", score)

if __name__ == "__main__":
    main()
