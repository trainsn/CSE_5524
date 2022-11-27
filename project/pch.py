import numpy as np
from skimage import color
from utils import *

import pdb

def color_histogram(img, nbins):
    num_channels = 3
    hist = np.zeros(nbins * num_channels, dtype=np.int)
    for i in range(num_channels):
        hist[i * nbins: (i + 1) * nbins] = np.histogram(img[:, :, i], bins=nbins, range=(0., 1.))[0]

    return hist

def compute_pch(img, nbins=10, levels=2):
    hsv = color.rgb2hsv(img)
    rgb = img / 255.
    height, width = img.shape[:2]
    num_channels = 6

    coef = 1
    hist_size = 0
    for k in range(levels + 1):
        hist_size += nbins * num_channels * coef
        coef *= 4

    # Creating the descriptor.
    hist = np.zeros(hist_size, dtype=np.int)

    blocks = 1
    binPos = 0  # Next free section in the histogram
    for k in range(levels + 1):
        wstep = width // blocks
        hstep = height // blocks
        for i in range(blocks):
            for j in range(blocks):
                hist[nbins * num_channels // 2 * binPos:nbins * num_channels // 2 * (binPos + 1)] = \
                    color_histogram(rgb[i * hstep: (i+1) * hstep, j * wstep: (j+1) * wstep], nbins)
                binPos += 1
                hist[nbins * num_channels // 2 * binPos:nbins * num_channels // 2 * (binPos + 1)] = \
                    color_histogram(hsv[i * hstep: (i + 1) * hstep, j * wstep: (j + 1) * wstep], nbins)
                binPos += 1
        blocks *= 2

    hist = l2_normalize(hist)

    return hist
