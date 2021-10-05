# Editor Name: Daniel Cheng
# Edit Time: 15:41 2021/9/26

from scipy.io import loadmat
from scipy.linalg import norm, pinv, inv
import numpy as np
from sklearn.cluster import KMeans


class RBF:

    def __init__(self, indim, num_centers, sigma, outdim):
        self.indim = indim
        self.num_centers = num_centers
        self.sigma = sigma
        self.outdim = outdim
        self.centers = [np.random.uniform(-1, 1, indim) for i in range(num_centers)]
        self.W = np.random.random((self.num_centers, self.outdim))

        # RBF Gaussian function
        
    def _basisfunc(self, xk, d):
        assert len(d) == self.indim
        return np.exp((-1 / (2 * self.sigma ** 2)) * norm(xk - d) ** 2)

    # calculate activations of RBFs
    def _calcAct(self, X):
        G = np.zeros((X.shape[0], self.num_centers), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    # X: input dimensions 330 x 33; Y: output column vector dimension 330 x 1
    def train(self, X, Y):

        # # i. random centers selection from training set
        # rnd_idx = np.random.permutation(X.shape[0])[:self.num_centers]
        # self.centers = [X[i, :] for i in rnd_idx]
        # print("center selected randomly:", self.centers)

        # ii. K means cluster to determine centers here
        model = KMeans(n_clusters=self.num_centers, max_iter=1000)
        model.fit(data_train)
        self.centers = model.cluster_centers_
        print("centers selected by K Means clustering:\n", self.centers)

        # calculate activations of RBFs
        G = self._calcAct(X)
        print(G)

        # # calculate output weights (pseudoinverse)
        # self.W = np.dot(pinv(G), Y)

        # calculate output weights (linear least square estimate)
        GT_G = np.dot(np.transpose(G), G)
        self.W = inv(GT_G) @ np.transpose(G) @ Y
        print('Output weights W is:\n', self.W)

        # X: input dimensions 330 x 33

    def test(self, X):
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y


if __name__ == '__main__':

#     file1 = 'C:\\Users\\程泽\\Desktop\\Continuous Assessment\\data_train.mat'
#     file2 = 'C:\\Users\\程泽\\Desktop\\Continuous Assessment\\label_train.mat'
#     file3 = 'C:\\Users\\程泽\\Desktop\\Continuous Assessment\\data_test.mat'
#     data_train = loadmat(file1, mat_dtype=True)['data_train']
#     label_train = loadmat(file2, mat_dtype=True)['label_train']
#     data_test = loadmat(file3, mat_dtype=True)['data_test']

    PATH = 'Data/'
    data_train = loadmat(PATH + 'data_train.mat')['data_train']
    label_train = loadmat(PATH + 'label_train.mat')['label_train']
    data_test = loadmat(PATH + 'data_test.mat')['data_test']

    # separate data set as validation data set/training data set (20/310)
    data_x_val = data_train[0:20, :]
    data_x_train = data_train[20:329, :]
    label_x_val = label_train[0:20, :]
    label_x_train = label_train[20:329, :]
    # print('label_x_val is:\n', np.transpose(label_x_val))

    # RBF training process
    # indim = 33, num_centers = 15, sigma = 2, outdim = 1
    rbf = RBF(33, 15, 2, 1)
    rbf.train(data_x_train, label_x_train)

    # RBF testing process
    test_result = rbf.test(data_test)
    print('the output of test data is:\n', test_result)
    # print result as 1/-1
    for res in test_result:
        print('1') if res > 0 else print('-1')

    # # validation accuracy comparison
    # val_test_result = rbf.test(data_x_val)
    # print('the output of validation data is:\n', val_test_result)
    # # print result as 1/-1
    # for rel in val_test_result:
    #     print('1') if rel > 0 else print('-1')
