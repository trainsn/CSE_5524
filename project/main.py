import pdb

import os
import argparse
import numpy as np
from skimage.io import imread
from pch import *
from phog import *

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('img_path', help='Path to img')
    parser.add_argument('--color-bins', type=int, default=10, help='Number of color bins')
    parser.add_argument('--color-hist-levels', type=int, default=2, help='Number of levels for color histogram')
    parser.add_argument('--orient-bins', type=int, default=60, help='Number of orientation bins')
    parser.add_argument('--phog-levels', type=int, default=2, help='Number of levels for PHOG descriptor')
    args = parser.parse_args()

    img = imread(os.path.join("UCMerced_LandUse", "Images", "storagetanks", "storagetanks00.tif"))

    color_pch = compute_pch(img, nbins=args.color_bins, levels=args.color_hist_levels)
    # color_pcm = color_moment(img)

    texture_phog = compute_phog(img, nbins=args.orient_bins, levels=args.phog_levels)
    pdb.set_trace()

if __name__ == '__main__':
    main()