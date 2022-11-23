import pdb

import os
import numpy as np
from skimage.io import imread
from color_feature import *

img = imread(os.path.join("UCMerced_LandUse", "Images", "storagetanks", "storagetanks00.tif"))

color_hist = color_histogram(img)
color_mome = color_moment(img)


