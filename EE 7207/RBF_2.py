# Editor Name: Daniel Cheng
# Edit Time: 15:41 2021/9/26

from scipy.io import loadmat
from scipy.linalg import norm, pinv, inv
import numpy
from sklearn.cluster import KMeans


class RBF:

    def __init__(self, indim, num_centers, sigma, outdim):
        self.indim = indim
        self.num_centers = num_centers
        self.sigma = sigma
        self.outdim = outdim
        self.centers = [numpy.random.uniform(-1, 1, indim) for i in range(num_centers)]
        self.W = numpy.random.random((self.num_centers, self.outdim))

        # RBF Gaussian function

    def _basisfunc(self, xk, d):
        assert len(d) == self.indim
        return numpy.exp((-1 / (2 * self.sigma ** 2)) * norm(xk - d) ** 2)

    # calculate activations of RBFs
    def _calcAct(self, X):
        G = numpy.zeros((X.shape[0], self.num_centers), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    # X: input dimensions 330 x 33; Y: output column vector dimension 330 x 1
    def train(self, X, Y):

        # # i. random centers selection from training set
        # rnd_idx = numpy.random.permutation(X.shape[0])[:self.num_centers]
        # self.centers = [X[i, :] for i in rnd_idx]
        # print("center selected randomly:", self.centers)

        # ii. K means cluster to determine centers here
        model = KMeans(n_clusters=self.num_centers, max_iter=1500)
        model.fit(data_train)
        self.centers = model.cluster_centers_
        print("centers selected by K Means clustering:\n", self.centers)

        # calculate activations of RBFs
        G = self._calcAct(X)
        print(G)

        # # calculate output weights (pseudoinverse)
        # self.W = numpy.dot(pinv(G), Y)

        # calculate output weights (linear least square estimate)
        gtrandot_g = numpy.dot(numpy.transpose(G), G)
        self.W = inv(gtrandot_g) @ numpy.transpose(G) @ Y
        print('Output weights W is:\n', self.W)

        # X: input dimensions 330 x 33

    def test(self, X):
        G = self._calcAct(X)
        Y = numpy.dot(G, self.W)
        return Y


if __name__ == '__main__':

    file1 = 'C:\\Users\\程泽\\Desktop\\Continuous Assessment\\data_train.mat'
    file2 = 'C:\\Users\\程泽\\Desktop\\Continuous Assessment\\label_train.mat'
    file3 = 'C:\\Users\\程泽\\Desktop\\Continuous Assessment\\data_test.mat'

    data_train = loadmat(file1, mat_dtype=True)['data_train']
    label_train = loadmat(file2, mat_dtype=True)['label_train']
    data_test = loadmat(file3, mat_dtype=True)['data_test']

    # RBF train
    # indim = 33, num_centers = 10, sigma = 0.25, outdim = 1
    rbf = RBF(33, 10, 0.25, 1)
    rbf.train(data_train, label_train)
    test_result = rbf.test(data_test)
    print('the output of test data is:\n', test_result)
    # print result
    for res in test_result:
        print('1') if res > 0 else print('-1')
