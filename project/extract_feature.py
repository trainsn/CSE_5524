import pdb

import os
import argparse
import numpy as np
from skimage.io import imread
from pch import *
from phog import *
from color_moment import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--color-bins', type=int, default=10, help='Number of color bins')
    parser.add_argument('--orient-bins', type=int, default=60, help='Number of orientation bins')
    args = parser.parse_args()

    classes = ["agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral", "denseresidential",
               "forest", "freeway", "golfcourse", "harbor", "intersection", "mediumresidential", "mobilehomepark",
               "overpass", "parkinglot", "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"]
    n_classes = len(classes)
    size_per_class = 100

    for i in range(n_classes):
        for j in range(size_per_class):
            img_name = os.path.join("UCMerced_LandUse", "Images", classes[i], "{}{:02d}.tif".format(classes[i], j))
            img = imread(img_name)

            save_dict = os.path.join("UCMerced_LandUse", "Features")
            for k in range(3):
                color_pch = compute_pch(img, nbins=args.color_bins, levels=k)
                np.save(os.path.join(save_dict, "pch", "level{:d}".format(k), classes[i], "{}{:02d}".format(classes[i], j)),
                        color_pch)

            color_mome = compute_color_moment(img)
            np.save(os.path.join(save_dict, "cm", classes[i], "{}{:02d}".format(classes[i], j)), color_mome)

            for k in range(3):
                texture_phog = compute_phog(img, nbins=args.orient_bins, levels=k)
                np.save(os.path.join(save_dict, "phog", "level{:d}".format(k), classes[i], "{}{:02d}".format(classes[i], j)),
                        texture_phog)

        print("finish processing class {}".format(classes[i]))

if __name__ == '__main__':
    main()
