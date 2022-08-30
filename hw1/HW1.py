import pdb
from PIL import Image, ImageOps # Importing Image and ImageOps module from PIL package
import numpy as np

grayIm = None

def task1():
    global rgbIm
    grayIm = Image.open("buckeyes_gray.bmp")
    pdb.set_trace()
    grayIm.show()
    grayIm.save("buckeyes_gray.jpg")

    rgbIm = Image.open('buckeyes_rgb.bmp')
    rgbIm.show()
    rgbIm.save("buckeyes_rgb.jpg")

def task2():
    global rgbIm
    grayIm = ImageOps.grayscale(rgbIm)
    grayIm.show()

def task3():
    zBlock = np.zeros((10, 10), dtype=np.uint8)
    oBlock = np.ones((10, 10), dtype=np.uint8) * 255
    pattern = np.vstack((np.hstack((zBlock, oBlock)), np.hstack((oBlock, zBlock))))
    checkerIm = np.tile(pattern, (5, 5))
    im = Image.fromarray(checkerIm)
    im.save("checkerIm.bmp")
    im = Image.open('checkerIm.bmp')
    im.show()

task1()
task2()
task3()
