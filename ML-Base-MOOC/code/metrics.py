import numpy as np
from math import sqrt

def accuracy_score(y_true, y_predict):
    """计算 y_true 和 y_predict 之间的准确率"""
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict) / len(y_true)

def mean_squared_error(y_true, y_predict):
    # 计算 MSE
    assert len(y_true) == len(y_predict)
    return np.sum((y_true - y_predict)**2) / len(y_true)

def root_mean_squared_error(y_true, y_predict):
    # 计算 RMSE
    return sqrt(mean_squared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    # 计算 RAE
    assert len(y_true) == len(y_predict), \
        "The size of y_true must be equal to the size of y_predict"

    return np.sum(np.abs(y_predict - y_true)) / len(y_true)

def r2_score(y_true, y_predict):
    # 计算 y_true 和 y_predict 之间的 R Square
    # 1 - （MSE / 方差）
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)

def TN(y_ture, y_predict):
    assert len(y_ture) == len(y_predict)
    return np.sum((y_ture == 0) & (y_predict == 0))

def FN(y_ture, y_predict):
    assert len(y_ture) == len(y_predict)
    return np.sum((y_ture == 1) & (y_predict == 0))

def FP(y_ture, y_predict):
    assert len(y_ture) == len(y_predict)
    return np.sum((y_ture == 0) & (y_predict == 1))

def TP(y_ture, y_predict):
    assert len(y_ture) == len(y_predict)
    return np.sum((y_ture == 1) & (y_predict == 1))

def precision_score(y_ture, y_predict):
    tp = TP(y_ture, y_predict)
    fp = FP(y_ture, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0

def recall_score(y_ture, y_predict):
    tp = TP(y_ture, y_predict)
    fn = FN(y_ture, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0

def f1_score(precision, recall):
    try:
        return (2 * precision * recall) / (precision + recall)
    except:
        return 0.0

def TPR(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0

def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.0