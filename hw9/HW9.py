import numpy as np

import pdb

P, x, X = None, None, None

def task1():
    global P, x, X
    x = np.loadtxt('2Dpoints.txt')
    X = np.loadtxt('3Dpoints.txt')
    N = x.shape[0]
    A = np.zeros((2 * N, 12))
    A[::2, :3] = X
    A[::2, 3] = 1.
    A[1::2, 4:7] = X
    A[1::2, 7] = 1.
    A[::2, 8:11] = -X * x[:, 0:1]
    A[::2, 11] = -x[:, 0]
    A[1::2, 8:11] = -X * x[:, 1:2]
    A[1::2, 11] = -x[:, 1]

    ew, ev = np.linalg.eig(A.T @ A)
    p = ev[:, np.argmin(ew)]
    p /= np.linalg.norm(p)
    P = p.reshape((3, 4))
    print("the camera matrix P is:")
    print(P)

def task2():
    global P, x, X
    N = x.shape[0]
    pdb.set_trace()
    X_homo = np.hstack((X, np.ones((N, 1))))
    x_pred_homo = (P @ X_homo.T).T
    x_pred = x_pred_homo[:, :2] / x_pred_homo[:, 2:]
    err = ((x_pred - x) ** 2).sum()
    print("sum-of-squared error is {:.4f}".format(err))

def task3():


# task1()
# task2()