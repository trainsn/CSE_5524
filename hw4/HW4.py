import os
import numpy as np
import imageio
import matplotlib.pyplot as plt

import pdb

Y = None
v = None
Y_prime = None

def similitudeMoments(im):
    Nvals = []
    ny, nx = im.shape
    y, x  = np.arange(ny), np.arange(nx)
    yv, xv = np.meshgrid(y, x, indexing='ij')
    m00 = np.sum(im)
    m10, m01 = np.sum(xv * im), np.sum(yv * im)
    x_bar, y_bar = m10 / m00, m01 / m00
    for i in range(4):
        for iplusj in range(2, 4):
            j = iplusj - i
            if j < 0:
                continue
            # print(i, j)
            eta = np.sum((xv - x_bar)**i * (yv - y_bar)**j * im) / (np.sum(im) ** ((i+j)/2. + 1.))
            Nvals.append(eta)
    return Nvals

def task1():
    for i in range(4):
        im = imageio.imread(os.path.join("HW4", "boxIm{:d}.bmp".format(i + 1)))
        im = im.astype(np.float32) / 255.
        Nvals = similitudeMoments(im)
        print(Nvals)

def task2():
    global Y
    X = np.loadtxt(os.path.join("HW4", "eigdata.txt"), delimiter=",")
    m = np.mean(X)
    Y = X - np.ones((X.shape[0], 1)) * m

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(X[:, 0], X[:, 1], "b.")
    axs[0].axis('equal')
    axs[0].set_title("X")
    axs[1].plot(Y[:, 0], Y[:, 1], "b.")
    axs[1].axis('equal')
    axs[1].set_title("Y")
    plt.subplots_adjust(wspace=0, hspace=0.4)
    plt.show()

def task3():
    global Y, v
    N, dim = Y.shape
    B = Y - Y.mean(axis=0)
    cov = B.T.dot(B) / N
    w, v = np.linalg.eig(cov)
    C = 9.
    v_prime = np.sqrt(w) * np.sqrt(C) * v
    plt.plot(Y[:, 0], Y[:, 1], "b.")
    m = Y.mean(axis=0)
    plt.arrow(m[0], m[1], v_prime[0, 0], v_prime[1, 0], head_width=0.2)
    plt.arrow(m[0], m[1], v_prime[0, 1], v_prime[1, 1], head_width=0.2)
    plt.axis('equal')
    plt.show()

def task4():
    global Y, v, Y_prime
    Y_prime = Y.dot(v)
    plt.plot(Y_prime[:, 0], Y_prime[:, 1], "b.")
    plt.axis('equal')
    plt.show()

def task5():
    global Y_prime
    plt.hist(Y_prime[:, 0], 10)
    plt.show()

task1()
task2()
task3()
task4()
task5()