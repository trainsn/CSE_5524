import numpy as np
import math
from skimage.feature import canny
from skimage.filters import sobel_h, sobel_v
from skimage import io
from utils import *
from matplotlib import pyplot as plt

import pdb

PI_OVER_TWO = np.pi / 2.0

def getHistogram(edges, ors, mag, startX, startY, width, height, nbins):
    hist = np.zeros(nbins)
    for y in range(startY, startY + height):
        for x in range(startX, startX + width):
            if edges[y, x] > 0:
                bin = math.floor(ors[y, x])
                if bin == nbins:
                    bin -= 1
                hist[bin] += mag[y, x]
    return hist

def compute_phog(img, nbins, levels):
    # io.imshow(img)
    # plt.show()
    height, width = img.shape[:2]

    # Determine desc size
    coef = 1
    desc_size = 0
    for k in range(levels + 1):
        desc_size += nbins * coef
        coef *= 4

    # Convert the image to grayscale
    if img.shape[2] == 3:
        img = (img[:, :, 0] * 0.3 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.11) / 255.
    # io.imshow(img)
    # plt.show()

    # Apply Canny Edge Detector
    mean = np.mean(img)
    edges = canny(img,
        low_threshold=0.66 * mean,
        high_threshold=1.33 * mean).astype('int') * 255
    # io.imshow(edges)
    # plt.show()

    # Computing the gradients.
    grad_x = sobel_h(img)
    grad_y = sobel_v(img)
    # io.imshow(grad_x)
    # plt.show()
    # io.imshow(grad_y)
    # plt.show()

    # Total Gradient
    grad_m = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # io.imshow(grad_m)
    # plt.show()

    # Computing orientations
    grad_o = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            if grad_x[y, x] != 0.0:
                grad_o[y, x] = math.atan(grad_y[y, x] / grad_x[y, x])
            else:
                grad_o[y, x] = PI_OVER_TWO
    # io.imshow(grad_o)
    # plt.show()

    # Quantizing orientations into bins.
    grad_o = (grad_o / np.pi + 0.5) * nbins

    # Creating the descriptor.
    desc = np.zeros(desc_size, dtype=np.float32)

    blocks = 1
    binPos = 0 # Next free section in the histogram
    for k in range(levels + 1):
        wstep = width // blocks
        hstep = height // blocks
        for i in range(blocks):
            for j in range(blocks):
                desc[nbins * binPos:nbins * (binPos + 1)] = \
                    getHistogram(edges, grad_o, grad_m, i * wstep, j * hstep, wstep, hstep, nbins)
                binPos += 1
        blocks *= 2

    desc = l2_normalize(desc)

    return desc
