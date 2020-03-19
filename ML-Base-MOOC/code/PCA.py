import numpy as np
import matplotlib.pyplot as plt

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        assert self.n_components <= X.shape[1], \
            "n_components must not be greater than the feature number of X"
        def demean(X):
            return X - np.mean(X, axis=0)

        def f(X, w):
            return np.sum((X.dot(w))**2) / len(X)

        def df(X, w):
            return X.T.dot(X.dot(w)) / 2. * len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):

            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(X, w)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if abs(f(X, w) - f(X, last_w)) < epsilon:
                    break
                cur_iter += 1
            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta, n_iters)
            self.components_[i, :] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
        return self

    def transform(self, X):
        """将给定的X，映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X，反向映射回原来的特征空间"""
        assert X.shape[1] == self.components_.shape[0]

        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components

def plt_image(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

def plt_all_data(X, X_restore):
    plt.scatter(X[:, 0], X[:, 1], color='b', alpha=0.5)
    plt.scatter(X_restore[:, 0], X_restore[:, 1], color='r', alpha=0.5)
    plt.show()

def main():
    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0., 100., size=100)
    X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10, size=100)
    plt_image(X)
    pca = PCA(n_components=1)
    pca.fit(X)
    print(pca.components_)
    X_reduction = pca.transform(X)
    X_restore = pca.inverse_transform(X_reduction)
    plt_all_data(X, X_restore)

if __name__ == '__main__':
    main()