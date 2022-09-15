import os
import numpy as np
from PIL import Image, ImageFilter # Importing Image and ImageOps module from PIL package
import imageio
from scipy import signal, misc
from skimage import measure

import pdb

d_bsIm = None

def task1():
    nc, nr = 32, 24
    c, r = nc * 8 + 1, nr * 8 + 1
    im = imageio.imread(os.path.join("task1", "jin.jpg"))
    im = np.array(Image.fromarray(im).resize((c, r)))   # preprocessing
    # imageio.imwrite("jin_resize.jpg", im)
    im_raw = im

    # blurs and samples the input, interpolates the blurred and sampled input to estimate the original
    a = 0.4
    w = np.array([.25 - .5 * a, .25, a, .25, .25 - .5 * a]).astype(np.float32)
    Laplacians = []
    for i in range(3):
        im_blur = signal.sepfir2d(im, w, w)
        im_sample = im_blur[::2, ::2].astype(np.uint8)
        imageio.imwrite(os.path.join("task1","jin_Gaussian{:d}.jpg".format(i + 1)), im_sample)
        im_interp = np.zeros_like(im_blur)
        col, row = im_blur.shape
        im_sample = im_sample.astype(np.float32)
        for j in range(col):
            for k in range(row):
                im_interp[j, k] = (im_sample[j//2, k//2] + im_sample[(j+1)//2, k//2] +
                                   im_sample[j//2, (k+1)//2] + im_sample[(j+1)//2, (k+1)//2]) / 4.
        im_diff = im - im_interp
        Laplacians.append(im_diff)
        imageio.imwrite(os.path.join("task1","jin_Laplacian{:d}.jpg".format(i + 1)),
                        im_diff) # Lossy conversion from float32 to uint8.
        im = im_sample

    # reconstruction
    for i in range(2, -1, -1):
        im_sample = im
        col, row = (im_sample.shape[0] - 1) * 2 + 1, (im_sample.shape[1] - 1) * 2 + 1
        im_interp = np.zeros((col, row))
        for j in range(col):
            for k in range(row):
                im_interp[j, k] = (im_sample[j // 2, k // 2] + im_sample[(j + 1) // 2, k // 2] +
                                   im_sample[j // 2, (k + 1) // 2] + im_sample[(j + 1) // 2, (k + 1) // 2]) / 4.
        im_diff = Laplacians[i]
        im = im_diff + im_interp
    imageio.imwrite(os.path.join("task1","jin_reconstruct.jpg"), im)
    print(abs(im - im_raw).sum())

def task2():
    im_walk = imageio.imread(os.path.join("HW3","walk.bmp")).astype(np.float64)
    im_bg = imageio.imread(os.path.join("HW3","bg000.bmp")).astype(np.float64)
    im_diff = abs(im_walk - im_bg)
    for T in range(20, 101, 20):
        im_object = (im_diff > T).astype(np.uint8) * 255
        imageio.imwrite(os.path.join("task2", "walk_T{:.0f}.jpg".format(T)), im_object)

def task3():
    im_walk = imageio.imread(os.path.join("HW3", "walk.bmp")).astype(np.float64)
    num_imgs = 30
    im_bgs = np.zeros((num_imgs, im_walk.shape[0], im_walk.shape[1]))
    for i in range(num_imgs):
        im_bgs[i] = imageio.imread(os.path.join("HW3", "bg{}.bmp".format(str(i).zfill(3)))).astype(np.float64)
    mu = im_bgs.mean(axis=0)
    sigma = im_bgs.std(axis=0)
    im_diff = ((im_walk - mu) ** 2) / (sigma ** 2)
    im_diff[sigma < 1e-4] = 0.
    for T in range(6, 31, 6):
        im_object = (im_diff > T ** 2).astype(np.uint8) * 255
        imageio.imwrite(os.path.join("task3", "walk_T{:.0f}.bmp".format(T)), im_object)

def task4():
    global d_bsIm
    bsIm = Image.open(os.path.join("task3", "walk_T18.bmp"))
    d_bsIm = bsIm.filter(ImageFilter.MaxFilter(3))
    d_bsIm.save(os.path.join("task4", "walk_bestDilate.bmp"))

def task5():
    blobs = measure.label(np.array(d_bsIm))
    max_region_size, max_region_id = 0, 0
    for i in range(blobs.max()):
        region_size = (blobs == i+1).sum()
        if region_size > max_region_size:
            max_region_size, max_region_id = region_size, i+1
    blobs[blobs != max_region_id] = 0
    imageio.imwrite(os.path.join("task5", "walk_largest_connected_region.bmp"), blobs)

task1()
task2()
task3()
task4()
task5()