import pdb

import os
import argparse
import numpy as np
from skimage.io import imread
from color_feature import *
from phog import *

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('img_path', help='Path to img')
    parser.add_argument('--nbins', type=int, default=60, help='Number of orientation bins')
    args = parser.parse_args()

    img = imread(os.path.join("UCMerced_LandUse", "Images", "storagetanks", "storagetanks00.tif"))

    color_hist = color_histogram(img)
    color_mome = color_moment(img)

    texture_phog = compute_phog(img, nbins=args.nbins)
    pdb.set_trace()

if __name__ == '__main__':
    main()