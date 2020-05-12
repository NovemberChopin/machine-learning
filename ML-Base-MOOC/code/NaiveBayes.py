"""
朴素贝叶斯算法的实现
2020/5/10
"""
import numpy as np
import pandas as pd
from model_slection import train_test_splice
from sklearn import datasets

class NaiveBayes():
    def __init__(self, lambda_):
        '''
        初始化参数
        :param lambda_: 贝叶斯系数 取0时，即为极大似然估计
        '''
        self.lambda_ = lambda_
        self.y_types_count = None  # y的（类型：数量）
        self.y_types_proba = None  # y的（类型：概率）
        self.x_types_proba = dict()  # （xi 的编号,xi的取值，y的类型）：概率

    def fit(self, X_train, y_train):
        self.y_types = np.unique(y_train)  # y的所有取值类型

        X = pd.DataFrame(X_train)
        y = pd.DataFrame(y_train)
        # y的（类型：数量）统计
        self.y_types_count = y[0].value_counts()

        # y的（类型：概率）计算
        self.y_types_proba = (self.y_types_count + self.lambda_) / (y.shape[0] + len(self.y_types) * self.lambda_)

        # （xi 的编号,xi的取值，y的类型）：概率的计算
        for idx in X.columns:  # 遍历xi
            for j in self.y_types:  # 选取每一个y的类型
                p_x_y = X[(y == j).values][idx].value_counts()
                # 选择所有y==j为真的数据点的第idx个特征的值，并对这些值进行（类型：数量）统计
                for i in p_x_y.index:
                    # 计算（xi 的编号,xi的取值，y的类型）：概率
                    self.x_types_proba[(idx, i, j)] = (p_x_y[i] + self.lambda_) / (
                            self.y_types_count[j] + p_x_y.shape[0] * self.lambda_)

    def predict(self, X_new):
        res = []
        for y in self.y_types:  # 遍历y的可能取值
            p_y = self.y_types_proba[y]  # 计算y的先验概率P(Y=ck)
            p_xy = 1
            for idx, x in enumerate(X_new):
                p_xy *= self.x_types_proba[(idx, x, y)]  # 计算P(X=(x1,x2...xd)/Y=ck)
            res.append(p_y * p_xy)
        for i in range(len(self.y_types)):
            print("[{}]对应概率：{:.2%}".format(self.y_types[i], res[i]))
        # 返回最大后验概率对应的y值
        return self.y_types[np.argmax(res)]


class GaussianNB():
    '''高斯贝叶斯分类器'''
    def __init__(self, lambda_):
        '''
        初始化
        :param lambda_: 贝叶斯系数 取0时，即为极大似然估计
        '''
        self.lambda_ = lambda_
        self.y_types_count = None  # y的（类型：数量）
        self.y_types_proba = None  # y的（类型：概率）
        self.x_types_proba = dict()  # （xi 的编号, y的类型）：概率


    def _calculate_mu_sigma(self, feature):
        '''
        计算 mu 和 sigma
        :param feature: 某一个 y 的值对应的数据中某一个属性的所有数据
        :return: mu and sigma
        '''
        mu = np.mean(feature)
        sigma = np.std(feature)
        return (mu, sigma)

    def fit(self, X_train, y_train):
        self.y_types = np.unique(y_train)

        X = pd.DataFrame(X_train)
        y = pd.DataFrame(y_train)

        self.y_types_count = y[0].value_counts()
        # y的（类型：概率）计算
        self.y_types_proba = (self.y_types_count + self.lambda_) / (y.shape[0] + len(self.y_types) * self.lambda_)

        for idx in X.columns:
            for j in self.y_types:
                # 标签 j 对应的所有数据中 idx 列属性的数据
                p_x_y = X[(y == j).values][idx]
                # {(属性, 标签值): (mu, sigma)}
                self.x_types_proba[(idx, j)] = self._calculate_mu_sigma(p_x_y)
        # print(self.x_types_proba)

    def _prob_gaussian(self, mu, sigma, x):
        '''
        计算给定高斯分布下某一点的概率密度值
        :param mu: mu
        :param sigma: sigma
        :param x: 要计算概率密度的点
        :return: 概率密度
        '''
        return (1.0 / (sigma * np.sqrt(2 * np.pi)) *
                np.exp(- (x - mu) ** 2 / (2 * sigma ** 2)))

    # given mu and sigma , return Gaussian distribution probability for target_value
    def _get_xj_prob(self, mu_sigma, target_value):
        '''
        得到某一点在高斯分布下的概率密度值
        :param mu_sigma: mu 和 sigma 元组
        :param target_value: 要求概率密度的点
        :return:
        '''
        return self._prob_gaussian(mu_sigma[0], mu_sigma[1], target_value)

    def _predict(self, x_new):
        '''
        预测单个数据结果
        :param x_new: 要预测的数据
        :return: 预测结果
        '''
        res = []
        for y in self.y_types:  # 遍历y的可能取值
            p_y = self.y_types_proba[y]  # 计算y的先验概率P(Y=ck)
            p_xy = 1
            for idx, x in enumerate(x_new):
                mu_sigma = self.x_types_proba[(idx, y)]
                p_xy *= self._get_xj_prob(mu_sigma, x)
            res.append(p_y * p_xy)

        return self.y_types[np.argmax(res)]

    def predict(self, X_test):
        '''
        对多个数据进行预测
        :param X_test: 要预测的数据列表
        :return: 结果列表
        '''
        return [self._predict(x) for x in X_test]

    def score(self, X_test, y_test):
        '''
        计算预测得分
        :param X_test: 测试数据
        :param y_test: 测试数据的标签
        :return: 预测准确度
        '''
        y_predict = self.predict(X_test)
        return sum(y_test == y_predict) / len(y_test)



def main():
    # 对于离散数据例子
    # X_train = np.array([
    #     [1, "S"],
    #     [1, "M"],
    #     [1, "M"],
    #     [1, "S"],
    #     [1, "S"],
    #     [2, "S"],
    #     [2, "M"],
    #     [2, "M"],
    #     [2, "L"],
    #     [2, "L"],
    #     [3, "L"],
    #     [3, "M"],
    #     [3, "M"],
    #     [3, "L"],
    #     [3, "L"]
    # ])
    # y_train = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    # clf = NaiveBayes(lambda_=1)
    # clf.fit(X_train, y_train)
    # X_new = np.array([2, "S"])
    # y_predict = clf.predict(X_new)
    # print("{}被分类为:{}".format(X_new, y_predict))


    # 鸢尾花分类例子（对于连续值的处理）
    iris = datasets.load_iris()
    X = iris['data']
    y = iris['target']
    X_train, y_train, X_test, y_test = train_test_splice(X, y, seed=333)
    clf = GaussianNB(lambda_=1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('score:', score)

if __name__ == "__main__":
    main()
