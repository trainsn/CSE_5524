import os
import numpy as np
import imageio
from scipy import ndimage
import skimage.morphology
import matplotlib.pyplot as plt

import pdb
temporal_extent = 22

def task1():
    im_prev = imageio.imread(os.path.join("HW5", "aerobic-001.bmp"))
    for i in range(1, temporal_extent):
        im = imageio.imread(os.path.join("HW5", "aerobic-{}.bmp".format(str(i + 1).zfill(3))))
        for threshold in range(4, 21, 4):
            diff = abs(im - im_prev) >= threshold
            diff = ndimage.binary_opening(diff, structure=np.ones((3, 3)))
            diff = skimage.morphology.remove_small_objects(diff, min_size=55, connectivity=1, in_place=False)
            diff = diff.astype(np.uint8) * 255
            imageio.imwrite(os.path.join("task1", "threshold{:d}".format(threshold),
                                         "diff-{:03}.bmp".format(i + 1)), diff)
        im_prev = im

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

def task2():
    diffs = []
    for i in range(1, temporal_extent):
        diff = imageio.imread(os.path.join("task1", "threshold8", "diff-{}.bmp".format(str(i + 1).zfill(3))))
        diff = diff > 127.5
        diffs.append(diff)

    figure, axs = plt.subplots(5, 5)
    MEI = np.zeros_like(diffs[0]).astype(np.bool)
    for i in range(0, temporal_extent - 1):
        MEI = MEI + diffs[i]
        # imageio.imwrite(os.path.join("task2", "MEI-{:03}.bmp".format(i + 2)), MEI.astype(np.uint8) * 255)
        axs[i // 5, i % 5].set_title("MEI-{:03d}".format(i + 2))
        axs[i // 5, i % 5].axis("off")
        axs[i // 5, i % 5].imshow(MEI.astype(np.uint8) * 255, cmap="gray")
    for i in range(temporal_extent - 1, 25):
        axs[i // 5, i % 5].axis("off")
    plt.show()

    figure, axs = plt.subplots(5, 5)
    MHI_prev = np.zeros_like(diffs[0]).astype(np.float32)
    for i in range(0, temporal_extent - 1):
        # tau = float(i + 1) / (temporal_extent - 1)
        MHI = np.zeros_like(MHI_prev)
        MHI[MHI_prev > 0] = MHI_prev[MHI_prev > 0] - 1.
        MHI[diffs[i]] = temporal_extent
        MHI_prev = MHI
        # imageio.imwrite(os.path.join("task2", "MHI-{:03}.bmp".format(i + 2)), MHI)
        axs[i // 5, i % 5].set_title("MHI-{:03d}".format(i + 2))
        axs[i // 5, i % 5].axis("off")
        axs[i // 5, i % 5].imshow((MHI / temporal_extent * 255).astype(np.uint8), cmap="gray")
    for i in range(temporal_extent - 1, 25):
        axs[i // 5, i % 5].axis("off")
    plt.show()

    MEI_similitudeMoments = similitudeMoments(MEI.astype(np.float32))
    print("similitude moments of final MEI: ", MEI_similitudeMoments)
    MHI_similitudeMoments = similitudeMoments(MHI.astype(np.float32) / temporal_extent)
    print("similitude moments of final MHI: ", MHI_similitudeMoments)

def task3():
    box0 = np.zeros((101, 101), dtype=np.float32)
    box0[40:61, 6:27] = 1.
    imageio.imwrite(os.path.join("task3", "box0.bmp"), box0)
    box1 = np.zeros((101, 101), dtype=np.float32)
    box1[41:62, 7:28] = 1.
    imageio.imwrite(os.path.join("task3", "box1.bmp"), box1)

    fx = ndimage.filters.correlate(box1, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.)
    imageio.imwrite(os.path.join("task3", "fx.bmp"), fx)
    fy = ndimage.filters.correlate(box1, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.)
    imageio.imwrite(os.path.join("task3", "fy.bmp"), fy)
    ft = ndimage.filters.correlate(box1, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.) - \
         ndimage.filters.correlate(box0, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.)
    imageio.imwrite(os.path.join("task3", "ft.bmp"), ft)

    denom = fx ** 2 + fy ** 2
    denom[abs(denom) < 1e-4] = 1.
    u = -ft * fx / denom
    v = -ft * fy / denom

    margin = 5
    x = np.arange(0, 21 + margin * 2, 1).astype(np.float32) - .5
    y = np.arange(0, 21 + margin * 2, 1).astype(np.float32) - .5
    x, y = np.meshgrid(x, y)

    pdb.set_trace()
    plt.figure(figsize=(10, 10))
    plt.imshow(box0[40-margin:61+margin, 6-margin:27+margin], cmap="gray")
    plt.quiver(x, y,
               u[40-margin:61+margin, 6-margin:27+margin], -v[40-margin:61+margin, 6-margin:27+margin],
                color = 'lime', headwidth = 2, headlength = 2, scale = 45)
    plt.axis("off")
    plt.show()

task1()
task2()
task3()
