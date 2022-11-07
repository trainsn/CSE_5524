import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pdb

def task1():
    left = imageio.imread(os.path.join("input", "left.png"))
    right = imageio.imread(os.path.join("input", "right.png"))

    w, h = left.shape
    kernel_width = 11
    max_offset = 50
    disparity = np.zeros((w, h), dtype=np.uint8)

    for i in range(kernel_width // 2, w - kernel_width // 2):
        if i % 10 == 0:
            print("processing the {:d}th row".format(i))
        for j in range(kernel_width // 2, h - kernel_width // 2):
            left_window = left[i - kernel_width // 2:i + 1 + kernel_width // 2,
                               j - kernel_width // 2:j + 1 + kernel_width // 2]
            max_ncc = float("-inf")
            best_offset = -1
            for k in range(0, max_offset):
                if j - k - kernel_width // 2 < 0:
                    break
                right_window = right[i - kernel_width // 2:i + 1 + kernel_width // 2,
                               j - k - kernel_width // 2:j + 1 - k + kernel_width // 2]
                denominator = np.std(left_window, ddof=1) * np.std(right_window, ddof=1) * (kernel_width * kernel_width - 1)
                ncc = ((left_window - left_window.mean()) * (right_window - right_window.mean())).sum() / denominator \
                      if abs(denominator) > 1e-4 else 0
                if ncc > max_ncc:
                    max_ncc = ncc
                    best_offset = k

            disparity[i, j] = best_offset * 255 / max_offset

    imageio.imwrite("disparity.png", disparity)

def task2():
    train = np.loadtxt(os.path.join("input", "train.txt"))
    test = np.loadtxt(os.path.join("input", "test.txt"))
    train_data, train_label = train[:, :2], (train[:, -1] - 1).astype(np.bool)
    test_data, test_label = test[:, :2], (test[:, -1] - 1).astype(np.bool)

    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', dpi=150)
    idx = 0
    for k in [1, 5, 11, 15]:
        acc = 0
        pred = np.zeros_like(test_label, dtype=np.bool)
        for i in range(test_data.shape[0]):
            dist = ((test_data[i] - train_data) ** 2).sum(1)
            indices = np.argpartition(dist, k)[:k]
            pred[i] = train_label[indices].sum() > k // 2
        acc = (~np.logical_xor(pred, test_label)).sum() / test_data.shape[0]
        print("k={:d}, acc={:.4f}".format(k, acc))

        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        axarr[idx // 2, idx % 2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axarr[idx // 2, idx % 2].scatter(test_data[:, 0], test_data[:, 1], c=test_label, cmap=cmap_bold, s=1)
        axarr[idx // 2, idx % 2].scatter(test_data[np.logical_xor(pred, test_label), 0],
                                         test_data[np.logical_xor(pred, test_label), 1],
                                         c='#00FF00', edgecolor='k', cmap=cmap_bold, s=10)
        axarr[idx // 2, idx % 2].set_title("k={:d}, Acc={:.2f}%".format(k, acc * 100))

        idx += 1

    plt.show()

# task1()
task2()