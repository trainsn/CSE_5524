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

    pdb.set_trace()
    minPos = dists.argmin()
    minPos = minPos // (W - window_cols), minPos % (W - window_cols)
    plt.imshow(dists, cmap="gray")
    plt.show()
    imageio.imwrite("best_match.jpg", target[minPos[0]:minPos[0] + window_rows, minPos[1]:minPos[1] + window_cols])



task1()