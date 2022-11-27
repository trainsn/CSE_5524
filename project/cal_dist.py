import pdb

import os
import argparse
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat-type', type=str, required=True, help='raw, cm, pch, phog')
    parser.add_argument('--level', type=int, default=0, help='level for pch or phog')
    parser.add_argument('--dist-metric', type=str, required=True, help='SAD, SSD, NCC')
    args = parser.parse_args()
    print(args)

    classes = ["agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral", "denseresidential",
               "forest", "freeway", "golfcourse", "harbor", "intersection", "mediumresidential", "mobilehomepark",
               "overpass", "parkinglot", "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"]

    n_classes = len(classes)
    train_size, val_size, test_size = 80, 10, 10
    size_per_class = train_size + val_size + test_size
    feats = []

    if args.feat_type == "raw":
        feat_dir = os.path.join("UCMerced_LandUse", "Images")
        for i in range(n_classes):
            for j in range(size_per_class):
                feat_name = os.path.join(feat_dir, classes[i], "{}{:02d}.tif".format(classes[i], j))
                feat = imread(feat_name)
                if feat.shape[0] != 256 or feat.shape[1] != 256:
                    feat = resize(feat, (256, 256))
                feats.append(feat)
    elif args.feat_type == "pch" or args.feat_type == "phog" or args.feat_type == "cm":
        if args.feat_type == "pch" or args.feat_type == "phog":
            feat_dir = os.path.join("UCMerced_LandUse", "Features", args.feat_type, "level{:d}".format(args.level))
        else:
            feat_dir = os.path.join("UCMerced_LandUse", "Features", args.feat_type)
        for i in range(n_classes):
            for j in range(size_per_class):
                feat_name = os.path.join(feat_dir, classes[i], "{}{:02d}.npy".format(classes[i], j))
                feat = np.load(feat_name)
                feats.append(feat)

    rnd_idx = np.load("rnd_idx.npy")

    dist_val = np.zeros((n_classes * val_size, n_classes * train_size))
    for i in range(n_classes):
        for j in range(val_size):
            val_feat = feats[i * size_per_class + rnd_idx[train_size + j]]
            for p in range(n_classes):
                for q in range(train_size):
                    train_feat = feats[p * size_per_class + rnd_idx[q]]
                    if args.dist_metric == "SAD":
                        dist_val[i * val_size + j, p * train_size + q] = abs(val_feat - train_feat).sum()
                    elif args.dist_metric == "SSD":
                        dist_val[i * val_size + j, p * train_size + q] = ((val_feat - train_feat) ** 2).sum()
                    elif args.dist_metric == "NCC":
                        for k in range(3):
                            dist_val[i * val_size + j, p * train_size + q] += \
                                ((val_feat[:, :, k] - val_feat[:, :, k].mean()) * \
                                (train_feat[:, :, k] - train_feat[:, :, k].mean())).sum() / \
                                np.std(val_feat, ddof=1) * np.std(train_feat, ddof=1) * (feats[0].shape[0] * feats[0].shape[1] - 1)

        print("finish processing class {}".format(classes[i]))

    if args.dist_metric == "SAD":
        np.save(os.path.join(feat_dir, "val_SAD"), dist_val)
    elif args.dist_metric == "SSD":
        np.save(os.path.join(feat_dir, "val_SSD"), dist_val)
    elif args.dist_metric == "NCC":
        np.save(os.path.join(feat_dir, "val_NCC"), dist_val)

if __name__ == '__main__':
    main()
