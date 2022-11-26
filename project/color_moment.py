import numpy as np
from skimage import color
from utils import *

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
    return np.array(Nvals)

def compute_color_moment(img):
    num_channels = 6
    moment_items = 7
    moments = np.zeros(moment_items * num_channels)
    for i in range(3):
        moments[i * moment_items: (i + 1) * moment_items] = similitudeMoments(img[:, :, i])

    hsv = color.rgb2hsv(img)
    for i in range(3):
        moments[(i + 3) * moment_items: (i + 4) * moment_items] = similitudeMoments(hsv[:, :, i])

    moments = normalize(moments)
    return moments

