# Editor Name: Daniel Cheng
# Edit Time: 21:35 2021/11/4

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(precision=2)


# import package to calculate FTR
def cosine_method(X):
    R0 = cosine_similarity(np.mat(np.transpose(X)))
    print('\nThe result Matrix of cosine amplitude method is:\n', R0)
    return R0


# define a function to do compositions
def transfer(a, b):
    a, b = np.array(a), np.array(b)
    R1 = np.zeros_like(a.dot(b))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            empty = []
            for k in range(a.shape[1]):
                empty.append(min(a[i, k], b[k, j]))  # choose minimum element
            R1[i, j] = max(empty)  # choose maximum element
    return R1


# calculate FER in a loop when R1=R2
def ftr_fer(Y):
    R0 = cosine_method(Y)
    R1 = R0
    while True:
        R2 = R1
        R1 = transfer(R2, R2)
        if (R1 == R2).all():
            print('\nThe Fuzzy Equivalence Relation matrix is:\n', R1)
            return np.around(R1, decimals=2)
            break
        else:
            continue


# alpha_cut cluster by 0/1, using FER above-mentioned
def clustering(X, alpha):
    R1 = ftr_fer(X)
    R_a = np.zeros_like(R1, dtype=int)
    for i in range(R_a.shape[0]):
        for j in range(R_a.shape[1]):
            if R1[i, j] >= alpha:
                R_a[i, j] = 1
            else:
                R_a[i, j] = 0
    return R_a


def main():
    X = [[0.1, 0.0, 0.2, 0.8, 0.3, 0.0, 0.5, 0.6, 0.0, 0.1, 0.3, 0.1, 0.2, 0.2, 0.1, 0.2],
         [0.7, 0.5, 0.2, 0.1, 0.0, 0.4, 0.0, 0.3, 0.5, 0.6, 0.2, 0.5, 0.0, 0.6, 0.7, 0.4],
         [0.2, 0.5, 0.2, 0.0, 0.4, 0.0, 0.4, 0.0, 0.1, 0.0, 0.1, 0.4, 0.2, 0.1, 0.1, 0.2],
         [0.0, 0.0, 0.4, 0.1, 0.3, 0.6, 0.1, 0.1, 0.4, 0.3, 0.4, 0.0, 0.6, 0.1, 0.1, 0.2]]

    # # test matrix X
    # X = [[0.1, 0.8, 0.4, 0.1, 0.1],
    #      [0.2, 0.0, 0.4, 0.1, 0.1],
    #      [0.3, 0.1, 0.0, 0.4, 0.1],
    #      [0.2, 0.1, 0.1, 0.2, 0.1],
    #      [0.2, 0.0, 0.1, 0.2, 0.6]]

    print("\nThe alpha-cut (alpha=0.40) clustering matrix is: \n", clustering(X, alpha=0.40))
    print("\nThe alpha-cut (alpha=0.50) clustering matrix is: \n", clustering(X, alpha=0.50))
    print("\nThe alpha-cut (alpha=0.80) clustering matrix is: \n", clustering(X, alpha=0.80))
    print("\nThe alpha-cut (alpha=0.85) clustering matrix is: \n", clustering(X, alpha=0.85))
    print("\nThe alpha-cut (alpha=0.90) clustering matrix is: \n", clustering(X, alpha=0.90))


if __name__ == "__main__":
    main()
