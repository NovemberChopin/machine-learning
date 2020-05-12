"""
朴决策树的实现
2019/4/18
"""
import numpy as np
import pandas as pd
from collections import Counter
import math
import time
from collections import namedtuple
from sklearn import datasets
from model_slection import train_test_splice


class Node(namedtuple("Node", "children type content feature label")):
    """定义节点"""
    # 孩子节点、分类特征的取值、节点内容、节点分类特征、标签
    def __repr__(self):
        return str(tuple(self))


class DecisionTree():
    """
    决策树分类器 --- 离散数据
    接受数据格式： Numpy or DataFrame
    """

    def __init__(self, method="info_gain_ratio"):
        self.tree = None
        self.method = method

    def _experienc_entropy(self, X):
        """计算经验熵"""
        # 统计每个取值的出现频率
        x_types_prob = X[X.columns[0]].value_counts() / X.shape[0]
        # 计算经验熵
        x_experience_entropy = sum((-p * math.log(p, 2) for p in x_types_prob))
        return x_experience_entropy

    def _conditional_entropy(self, X_train, y_train, feature):
        """计算离散数据条件熵"""
        # feature特征下每个特征取值数量统计
        x_types_count = X_train[feature].value_counts()
        # 每个特征取值频率计算
        x_types_prob = x_types_count / X_train.shape[0]
        # 每个特征取值下类别y的经验熵
        x_experience_entropy = [self._experienc_entropy(y_train[(X_train[feature] == i)]) for i in
                               x_types_count.index]
        # 特征feature对数据集的经验条件熵
        x_conditional_entropy = (x_types_prob.values * x_experience_entropy).sum()
        return x_conditional_entropy

    def _information_gain(self, X_train, y_train, feature):
        """计算信息增益"""
        return self._experienc_entropy(y_train) - self._conditional_entropy(X_train, y_train, feature)

    def _information_gain_ratio(self, X_train, y_train, features, feature):
        """计算信息增益比"""
        index = features.index(feature)
        return self._information_gain(X_train, y_train, feature) / self._experienc_entropy(
            X_train.iloc[:, index:index + 1])

    def _choose_feature(self, X_train, y_train, features):
        """选择分类特征"""
        # 计算所有特征的信息增益
        if self.method == "info_gain_ratio":
            info = [self._information_gain_ratio(X_train, y_train, features, feature) for feature in features]
        elif self.method == "info_gain":
            info = [self._information_gain(X_train, y_train, feature) for feature in features]
        else:
            raise TypeError
        optimal_feature = features[np.argmax(info)]
        # for i in range(len(features)):
        #     print(features[i],":",info[i])
        return optimal_feature

    def _built_tree(self, X_train, y_train, features, type=None):
        '''
        递归构造决策树，递归停止的条件
            1、给定结点的所有样本属于同一类。
            2、没有剩余属性可以用来进一步划分样本。此时，使用多数表决，用样本中的多数所在的类标记它。
        :param type: 分类特征的取值
        :return: 构造好的决策树
        '''
        # 只有一个节点或已经完全分类，则决策树停止继续分叉
        if len(features) == 1 or len(np.unique(y_train)) == 1:
            label = list(y_train[0].value_counts().index)[0]
            return Node(children=None, type=type, content=(X_train, y_train), feature=None, label=label)
        else:
            # 选择分类特征值
            feature = self._choose_feature(X_train, y_train, features)
            features.remove(feature)
            # 构建节点，同时递归创建孩子节点
            features_iter = np.unique(X_train[feature])
            children = []
            for item in features_iter:
                X_item = X_train[(X_train[feature] == item)]
                y_item = y_train[(X_train[feature] == item)]
                children.append(self._built_tree(X_item, y_item, features, type=item))
            return Node(children=children, type=type, content=None, feature=feature, label=None)

    def _prune(self):
        """进行剪枝"""
        pass

    def fit(self, X_train, y_train):
        # 如果输入的数据不是 DataFrame 数据，进行转化
        # if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)

        # 特征数据转化为 List 数据
        features = X_train.columns.to_list()
        self.tree = self._built_tree(X_train, y_train, features)
        # self.tree=self._prune(tree)

    def _search(self, X_new):
        tree = self.tree
        # 若还有孩子节点，则继续向下搜索，否则搜索停止，在当前节点获取标签
        while tree.children:
            for child in tree.children:
                if X_new[tree.feature] == child.type:
                    tree = child
                    break
        return tree.label

    def predict(self, X_new):

        X_new = pd.DataFrame(X_new)
        return [self._search(X_new.loc[i]) for i in range(len(X_new))]

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return sum(y_test == y_predict) / len(y_test)

class Node2(namedtuple("Node", "lchild rchild value feature label")):
    """定义节点 --- 连续型数据"""
    # 左孩子节点、右孩子节点、划分节点值、分类特征、标签
    def __repr__(self):
        return str(tuple(self))

class DecisionTree2():
    """
    决策树分类器 --- 连续数据
    接受数据格式： Numpy
    """
    def __init__(self, depth):
        self.tree = None    # 构建的二叉树
        self.depth = depth  # 二叉树的深度

    def split(self, X_train, y_train, d, value):
        '''
        基于维度 d 的 value 值进行划分
        :param d: 要划分的维度
        :param value: 划分基准点
        :return: 分割后的两部分数据
        '''
        index_a = (X_train[:, d] <= value)
        index_b = (X_train[:, d] > value)
        return X_train[index_a], X_train[index_b], y_train[index_a], y_train[index_b]

    def entropy(self, y_train):
        '''
        计算 y_train 样本点的熵
        :return: 熵
        '''
        counter = Counter(y_train)
        res = 0.0
        for num in counter.values():
            p = num / len(y_train)
            res += -p * math.log(p)
        return res

    def try_split(self, X, y):
        '''
        寻找要划分的 value 值，寻找最小信息熵及相应的点
        :return:
        '''
        best_entropy = float('inf')  # 最小的熵的值
        best_d, best_v = -1, -1  # 划分的维度，划分的位置
        # 遍历每一个维度
        for d in range(X.shape[1]):
            # 每两个样本点在 d 这个维度中间的值. 首先把 d 维所有样本排序
            sorted_index = np.argsort(X[:, d])
            for i in range(1, len(X)):
                if X[sorted_index[i - 1], d] != X[sorted_index[i], d]:
                    v = (X[sorted_index[i - 1], d] + X[sorted_index[i], d]) / 2
                    x_l, x_r, y_l, y_r = self.split(X, y, d, v)
                    # 计算当前划分后的两部分结果熵是多少
                    e = self.entropy(y_l) + self.entropy(y_r)
                    if e < best_entropy:
                        best_entropy, best_d, best_v = e, d, v
        return best_entropy, best_d, best_v

    def build_tree(self, X_train, y_train):
        self.depth -= 1
        if self.depth < 0 or len(np.unique(y_train)) == 1:
            # 投票选出最多的标签作为叶子节点的 label
            label = Counter(y_train).most_common(1)[0][0]
            return Node2(lchild=None, rchild=None, value=None, feature=None, label=label)
        else:
            # 选择出分类特征， 分类节点值
            best_entropy, best_d, best_v = self.try_split(X_train, y_train)
            # 根据节点分割数据
            x_l, x_r, y_l, y_r = self.split(X_train, y_train, best_d, best_v)
            # 递归构建左右子树
            lchild = self.build_tree(x_l, y_l)
            rchild = self.build_tree(x_r, y_r)
            return Node2(lchild=lchild, rchild=rchild, value=best_v, feature=best_d, label=None)

    def fit(self, X_train, y_train):
        self.tree = self.build_tree(X_train, y_train)

    def _predict(self, x_train):
        tree = self.tree
        while tree.value:
            feature = tree.feature
            if tree.label:  # 如果当前是叶子节点
                break
            if x_train[feature] <= tree.value:
                tree = tree.lchild
            else:
                tree = tree.rchild
        return tree.label


    def predict(self, X_train):
        return [self._predict(x) for x in X_train]

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return sum(y_predict == y_test) / len(y_test)

def testIris():
    # 鸢尾花数据分类测试
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, y_train, X_test, y_test = train_test_splice(X, y, seed=333)
    clf = DecisionTree2(depth=4)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print(y_predict)
    print(y_test)
    score = clf.score(X_test, y_test)
    print("score: ", score)

def testData():
    star = time.time()
    # 训练数据集
    features = ["年龄", "有工作", "有自己的房子", "信贷情况"]
    X_train = np.array([
        ["青年", "否", "否", "一般"],
        ["青年", "否", "否", "好"],
        ["青年", "是", "否", "好"],
        ["青年", "是", "是", "一般"],
        ["青年", "否", "否", "一般"],
        ["中年", "否", "否", "一般"],
        ["中年", "否", "否", "好"],
        ["中年", "是", "是", "好"],
        ["中年", "否", "是", "非常好"],
        ["中年", "否", "是", "非常好"],
        ["老年", "否", "是", "非常好"],
        ["老年", "否", "是", "好"],
        ["老年", "是", "否", "好"],
        ["老年", "是", "否", "非常好"],
        ["老年", "否", "否", "一般"]
    ])
    y_train = np.array(["否", "否", "是", "是", "否", "否", "否", "是", "是", "是", "是", "是", "是", "是", "否"])
    # 转换成pd.DataFrame模式
    X_train = pd.DataFrame(X_train, columns=features)
    y_train = pd.DataFrame(y_train)
    # 训练
    clf = DecisionTree(method="info_gain")
    clf.fit(X_train, y_train)
    # 准备测试数据
    X_test = np.array([["青年", "是", "否", "一般"], ["中年", "是", "否", "好"]])
    y_test = np.array(["是", "是"])
    X_test = pd.DataFrame(X_test, columns=features)
    # 打印得分
    score = clf.score(X_test, y_test)
    print("score: ", score)
    print("time:{:.4f}s".format(time.time() - star))

def main():
    # testData()
    testIris()


if __name__ == "__main__":
    main()
