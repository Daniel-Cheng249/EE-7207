# Editor Name: Daniel Cheng
# Edit Time: 15:41 2021/9/26
# next step: k means or SOM NN to get centers/matrix splice to get a 30*30 validation set

from scipy.io import loadmat
from scipy.linalg import norm, pinv, inv
import numpy
from matplotlib import pyplot as plt


class RBF:

    def __init__(self, indim, num_centers, outdim):
        self.indim = indim
        self.outdim = outdim
        self.num_centers = num_centers
        self.centers = [numpy.random.uniform(-1, 1, indim) for i in range(num_centers)]
        self.sigma = 0.25
        self.W = numpy.random.random((self.num_centers, self.outdim))

    def _basisfunc(self, xk, d):
        # print(len(d),self.indim)
        assert len(d) == self.indim
        return numpy.exp((-1 / (2 * self.sigma ** 2)) * norm(xk - d) ** 2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = numpy.zeros((X.shape[0], self.num_centers), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G
    
        """ X: matrix of dimensions 330 x 33
            y: column vector of dimension 330 x 1 """
    def train(self, X, Y):
        # random center vectors from training set, SOM/K means was not used here
        rnd_idx = numpy.random.permutation(X.shape[0])[:self.num_centers]
        self.centers = [X[i, :] for i in rnd_idx]
        print("center", self.centers)

        # calculate activations of RBFs
        G = self._calcAct(X)
        print(G)

        # # calculate output weights (pseudoinverse)
        # self.W = numpy.dot(pinv(G), Y)

        # calculate output weights (linear least square estimate)
        Gtran_G = numpy.dot(numpy.transpose(G), G)
        self.W = inv(Gtran_G) @ numpy.transpose(G) @ Y
        print('Output weights W is:', self.W)

    def test(self, X):
        """ X: matrix of dimensions 330 x 33 """

        G = self._calcAct(X)
        Y = numpy.dot(G, self.W)
        return Y


if __name__ == '__main__':

      PATH = 'Data/'
      data_train = loadmat(PATH + 'data_train.mat')['data_train']
      label_train = loadmat(PATH + 'label_train.mat')['label_train']
      data_test = loadmat(PATH + 'data_test.mat')['data_test']    
    
#     file1 = 'C:\\Users\\程泽\\Desktop\\Continuous Assessment\\data_train.mat'
#     file2 = 'C:\\Users\\程泽\\Desktop\\Continuous Assessment\\label_train.mat'
#     file3 = 'C:\\Users\\程泽\\Desktop\\Continuous Assessment\\data_test.mat'
#     data_train = loadmat(file1, mat_dtype=True)
#     label_train = loadmat(file2, mat_dtype=True)
#     data_test = loadmat(file3, mat_dtype=True)

    # rbf train
    rbf = RBF(33, 30, 1)
    rbf.train(data_train, label_train)
    y_test = rbf.test(data_test)
    print('the output of test data is:', y_test)
    # print result
    for res in y_test:
        if res > 0:
            print('1')
        elif res < 0:
            print('-1')


