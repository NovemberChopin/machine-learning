import numpy as np
from math import sqrt

def accuracy_score(y_true, y_predict):
	"""计算 y_true 和 y_predict 之间的准确率"""
	assert y_true.shape[0] == y_predict.shape[0], \
		"the size of y_true must be equal to the size of y_predict"

	return sum(y_true == y_predict) / len(y_true)

def mean_squared_error(y_true, y_predict):
	# 计算 MSE
	assert len(y_true) == len(y_predict), \
		"The size of y_true must be equal to the size of y_predict"

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
