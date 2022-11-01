import numpy as np
from matplotlib import pyplot as plt

import pdb

def task1(x, X):
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

    return P

def task2(P, x, X):
    N = x.shape[0]
    X_homo = np.hstack((X, np.ones((N, 1))))
    x_pred_homo = (P @ X_homo.T).T
    x_pred = x_pred_homo[:, :2] / x_pred_homo[:, 2:]
    err = ((x_pred - x) ** 2).sum()
    print("sum-of-squared error is {:.4f}".format(err))

def computeTransform(im):
    x, y = im[:, 0], im[:, 1]
    s = np.sqrt(2) / np.sqrt((x - x.mean()) ** 2 + (y - y.mean()) ** 2).mean()
    T = np.zeros((3, 3))
    T[0, 0] = s
    T[0, 2] = -s * x.mean()
    T[1, 1] = s
    T[1, 2] = -s * y.mean()
    T[2, 2] = 1

    return T

def normalize(im, T):
    N = im.shape[0]
    im = np.hstack((im, np.ones((N, 1))))
    im = (T @ im.T).T

    return im

def task3(P1, P2):
    Ta = computeTransform(P1)
    Tb = computeTransform(P2)
    P1_norm = normalize(P1, Ta)
    P2_norm = normalize(P2, Tb)

    # computer H_tilde
    N = Ps.shape[0]
    A = np.zeros((2 * N, 9))
    A[::2, :3] = P1_norm
    A[1::2, 3:6] = P1_norm
    A[::2, 6:9] = -P1_norm * P2_norm[:, 0:1]
    A[1::2, 6:9] = -P1_norm * P2_norm[:, 1:2]

    ew, ev = np.linalg.eig(A.T @ A)
    h = ev[:, np.argmin(ew)]
    h /= np.linalg.norm(h)
    H_tilde = h.reshape((3, 3))

    # Compute H
    H = np.linalg.inv(Tb).dot(H_tilde).dot(Ta)
    print("the homography H is:")
    print(H)

    return H

def task4(H, P1, P2):
    N = P1.shape[0]
    P1_homo = np.hstack((P1, np.ones((N, 1))))
    P1_pred_homo = (H @ P1_homo.T).T
    P1_pred = P1_pred_homo[:, :2] / P1_pred_homo[:, 2:]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(P2[:, 0], P2[:, 1], alpha=0.8, c='red', edgecolors='none', s=45, label='Original', marker='x')
    ax.scatter(P1_pred[:, 0], P1_pred[:, 1], alpha=0.8, c='green', edgecolors='none', s=30, label='Projected', marker='D')
    plt.title('Points comparision')
    plt.legend(loc=2)
    plt.show()

    return P1_pred

def task5(P1_pred, P2):
    err = ((P1_pred - P2) ** 2).sum()
    print("sum-of-squared error is {:.4f}".format(err))

x = np.loadtxt('2Dpoints.txt')
X = np.loadtxt('3Dpoints.txt')
P = task1(x, X)
task2(P, x, X)

Ps = np.loadtxt("homography.txt", delimiter=',')
P1, P2 = Ps[:, :2], Ps[:, 2:]
H = task3(P1, P2)
P1_pred = task4(H, P1, P2)
task5(P1_pred, P2)