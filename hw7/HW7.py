import os
import numpy as np
import imageio
from skimage.segmentation import slic
from matplotlib import pyplot as plt

import pdb

def task1():
    img = imageio.imread("jin.jpg")
    for n_segments in [25, 50, 75, 100]:
        for compact in [1., 10., 25., 100.]:
            segments_slic = slic(img, n_segments=n_segments, compactness=compact)
            plt.clf()
            plt.imshow(segments_slic)
            plt.colorbar()
            plt.savefig(os.path.join("task1", "segment{:03d}_compactness{:.2f}.png".format(n_segments, compact)))

def task2():
    template = imageio.imread("template.png")
    search = imageio.imread("search.png")
    th, tw, _ = template.shape
    sh, sw, _ = search.shape
    nccs = np.zeros((sh - th + 1, sw - tw + 1), dtype=np.float32)
    for i in range(sh - th):
        if i % 10 == 0:
            print("processing row{:d}".format(i))
        for j in range(sw - tw):
            for k in range(3):
                patch = search[i:i+th, j:j+tw]
                nccs[i, j] += ((template[:, :, k] - template[:, :, k].mean()) *
                               (patch[:, :, k] - patch[:, :, k].mean())).sum() / \
                                (np.std(patch, ddof=1) * np.std(template, ddof=1) * (th * tw - 1))

    indices = (-nccs.flatten()).argsort()
    for i in [1, 2, 5, 10, 100, 500]:
        bh, bw = indices[i - 1] // nccs.shape[1], indices[i - 1] % nccs.shape[1]
        imageio.imwrite(os.path.join("task2", "best{:03d}.png".format(i)), search[bh:bh+th, bw:bw+tw])

    nccs = np.sort(nccs, axis=None)
    nccs = nccs[::-1]
    plt.clf()
    plt.plot(nccs)
    plt.title("Resulting Scores from Best to Worst")
    plt.savefig(os.path.join("task2", "1d-plot.png"))

task1()
task2()