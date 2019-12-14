import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed = None):
    if seed:
        np.random.seed(seed)

    shuffle_indexs = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)

    train_indexs = shuffle_indexs[test_size:]
    test_indexs = shuffle_indexs[:test_size]

    X_train = X[train_indexs]
    y_train = y[train_indexs]

    X_test = X[test_indexs]
    y_test = y[test_indexs]

    return X_train, X_test, y_train, y_test

def train_test_split2(X, y, test_ratio=0.2, seed = None):
    if seed:
        np.random.seed(seed)

    data = np.hstack([X, y.reshape(-1, 1)])
    shuffle_data = np.random.shuffle(data)
    X = shuffle_data[:, :4]
    y = shuffle_data[:, 4:]

    test_size = int(len(X) * test_ratio)

    X_train = X[test_size:]
    y_train = y[test_size:]

    X_test = X[:test_size]
    y_test = y[:test_size]

    return X_train, X_test, y_train, y_test
