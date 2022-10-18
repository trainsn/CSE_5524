import os
import numpy as np
import imageio
from scipy.linalg import eigh
from matplotlib import pyplot as plt

import pdb

def task1():
    model_cov = np.array([[47.917, 0, -146.636, -141.572, -123.269],
                                 [0, 408.250, 68.487, 69.828, 53.479],
                                 [-146.636, 68.487, 2654.285, 2621.672, 2440.381],
                                 [-141.572, 69.828, 2621.672, 2597.818, 2435.368],
                                 [-123.269, 53.479, 2440.381, 2435.368, 2404.923]])

    target = imageio.imread("target.jpg")
    H, W, _ = target.shape
    window_rows, window_cols = 70, 24
    y, x = np.arange(window_rows).astype(np.float32), np.arange(window_cols).astype(np.float32)
    yv, xv = np.meshgrid(y, x, indexing='ij')
    yv, xv  = yv.flatten()[np.newaxis, :], xv.flatten()[np.newaxis, :]
    R, G, B = target[:, :, 0].astype(np.float32), target[:, :, 1].astype(np.float32), target[:, :, 2].astype(np.float32)

    dists = np.zeros((H - window_rows, W - window_cols), dtype=np.float32)

    for i in range(H - window_rows):
        if i % 50 == 0:
            print("processing the {}th row".format(i))
        for j in range(W - window_cols):
            feat = np.vstack((xv, yv,
                              R[i:i+window_rows, j:j+window_cols].flatten()[np.newaxis, :],
                              G[i:i+window_rows, j:j+window_cols].flatten()[np.newaxis, :],
                              B[i:i+window_rows, j:j+window_cols].flatten()[np.newaxis, :]))
            candidate_cov = np.cov(feat, bias=True)
            eigvals = eigh(model_cov, candidate_cov, eigvals_only=True)
            dist = np.sqrt((np.log(eigvals)**2).sum())
            dists[i, j] = dist

    minPos = dists.argmin()
    minPos = minPos // (W - window_cols), minPos % (W - window_cols)
    plt.imshow(dists, cmap="gray")
    plt.show()
    print(minPos)
    imageio.imwrite("best_match.jpg", target[minPos[0]:minPos[0] + window_rows, minPos[1]:minPos[1] + window_cols])

def circularNeighbors(img, x, y, radius):
    H, W, _ = img.shape
    X = []
    for i in range(H):
        for j in range(W):
            if (i - y) ** 2 + (j - x) ** 2 < radius ** 2:
                X.append(np.array([j, i, img[i, j, 0], img[i, j, 1], img[i, j, 2]]))

    return np.array(X).astype(np.float32)

def colorHistogram(X, bins, x, y, h):
    hist = np.zeros((bins, bins, bins), dtype=np.float32)
    step = (255. + 1e-2) / bins
    for i in range(X.shape[0]):
        feat = X[i]
        r_idx, g_idx, b_idx = int(feat[2] // step), int(feat[3] // step), int(feat[4] // step)
        px, py = feat[0], feat[1]
        r = ((py - y) ** 2 + (px - x) ** 2) / (h ** 2)
        hist[r_idx, g_idx, b_idx] += 1 - r

    return hist / hist.sum()

def meanshiftWeights(X, q_model, p_test, bins):
    div = np.zeros((bins, bins, bins), dtype=np.float32)
    eps = 1e-4
    valid = p_test >= eps
    div[valid] = np.sqrt(q_model[valid] / p_test[valid])

    w = []
    step = (255. + 1e-2) / bins
    for i in range(X.shape[0]):
        feat = X[i]
        r_idx, g_idx, b_idx = int(feat[2] // step), int(feat[3] // step), int(feat[4] // step)
        w.append(div[r_idx, g_idx, b_idx])

    return np.array(w)

def task5():
    img1 = imageio.imread("img1.jpg")
    img2 = imageio.imread("img2.jpg")
    cx, cy = 150., 175.
    radius = 25.
    bins = 16
    num_iter = 25

    plt.imshow(img1)
    circle1 = plt.Circle((cx, cy), radius=25, color='b', fill=False)
    plt.gcf().gca().add_artist(circle1)
    plt.axis('off')
    plt.show()

    cNeigh = circularNeighbors(img1, cx, cy, radius)
    q_model = colorHistogram(cNeigh, bins, cx, cy, radius)

    old_loc = np.array([cx, cy])
    for i in range(num_iter):
        cNeigh2 = circularNeighbors(img2, old_loc[0], old_loc[1], radius)
        p_test = colorHistogram(cNeigh2, bins, old_loc[0], old_loc[1], radius)
        w = meanshiftWeights(cNeigh2, q_model, p_test, bins)
        new_loc = (w[:, np.newaxis] * cNeigh2[:, :2]).sum(axis=0) / w.sum()
        print(new_loc, np.sqrt(((new_loc - old_loc) ** 2).sum()))
        old_loc = new_loc

    plt.clf()
    plt.imshow(img2)
    circle2 = plt.Circle((new_loc[0], new_loc[1]), radius=25, color='b', fill=False)
    plt.gcf().gca().add_artist(circle2)
    plt.axis('off')
    plt.show()

task1()
task5()