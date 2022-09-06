import os
import math
import numpy as np
from PIL import Image, ImageOps, ImageFilter  # Importing Image and ImageFilter module from PIL package
import scipy.ndimage
import imageio
from skimage import feature
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pdb

sigma = 6.

def task1():
    # creating a image object
    faceIm = Image.open("affleck_gray.png")

    for i in range(40, 0, -3):
        sigma = i / 2.
        # applying the Gaussian Blur filter
        gIm = faceIm.filter(ImageFilter.GaussianBlur(radius=sigma))
        gIm.save(os.path.join("affleck_gaussian", "{:.1f}.bmp".format(sigma)))

def gaussDeriv2D(sigma):
    mask_size = math.ceil(3 * sigma) * 2 + 1    # 99.7% of weight
    Gx = np.zeros((mask_size, mask_size))
    Gy = np.zeros((mask_size, mask_size))
    for i in range(mask_size):
        for j in range(mask_size):
            x = i - mask_size // 2
            y = j - mask_size // 2
            Gx[i, j] = x / (2 * np.pi * np.power(sigma, 4.)) * np.exp(-(x * x + y * y) / (2 * sigma * sigma))
            Gy[i, j] = y / (2 * np.pi * np.power(sigma, 4.)) * np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    return Gx, Gy

def task2():
    mask_size = math.ceil(3 * sigma) * 2 + 1  # 99.7% of weight
    Gx, Gy = gaussDeriv2D(sigma)
    x = np.linspace(0, mask_size - 1, mask_size)
    y = np.linspace(0, mask_size - 1, mask_size)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Gx)
    ax.set_title('Gx')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, Gy)
    ax.set_title('Gy')

    plt.show()

def task3():
    lines = imageio.imread("lines.jpg").astype(np.float32)
    Gx, Gy = gaussDeriv2D(sigma)

    gxIm = scipy.ndimage.filters.convolve(lines, Gx, mode='nearest')
    imageio.imwrite("gxIm.jpg", abs(gxIm))
    gyIm = scipy.ndimage.filters.convolve(lines, Gy, mode='nearest')
    imageio.imwrite("gyIm.jpg", abs(gyIm))
    magIm = np.sqrt(gxIm * gxIm + gyIm * gyIm)
    imageio.imwrite("magIm.jpg", magIm)

def task4():
    magIm = imageio.imread("magIm.jpg")
    for T in range(50, 201, 50):
        tIm = (magIm > T) * 255.
        imageio.imwrite("Gaussian_tIm{:d}.jpg".format(T), tIm)

def task5():
    lines = imageio.imread("lines.jpg").astype(np.float32)
    fxIm = scipy.ndimage.sobel(lines, 0)  # horizontal derivative
    fyIm = scipy.ndimage.sobel(lines, 1)  # vertical derivative
    magIm = np.sqrt(fxIm * fxIm + fyIm * fyIm)
    magIm *= 255.0 / np.max(magIm)
    for T in range(50, 201, 50):
        tIm = (magIm > T) * 255.
        imageio.imwrite("Sobel_tIm{:d}.jpg".format(T), tIm)

def task6():
    lines = imageio.imread("lines.jpg").astype(np.float32)
    edges = feature.canny(lines) * 255.
    imageio.imwrite("canny.jpg", edges)

# task1()
# task2()
# task3()
# task4()
# task5()
task6()