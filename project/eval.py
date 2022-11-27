import os
import argparse
import numpy as np

import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-retrieved', type=int, default=12, help='Number of images retrieved')
    parser.add_argument('--feat-type', type=str, required=True, help='raw, cm, pch, phog')
    parser.add_argument('--level', type=int, default=0, help='level for pch or phog')
    parser.add_argument('--dist-metric', type=str, help='SAD, SSD, NCC')
    args = parser.parse_args()
    print(args)

    classes = ["agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral", "denseresidential",
               "forest", "freeway", "golfcourse", "harbor", "intersection", "mediumresidential", "mobilehomepark",
               "overpass", "parkinglot", "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"]

    n_classes = len(classes)
    train_size, val_size, test_size = 80, 10, 10
    size_per_class = train_size + val_size + test_size

    if args.feat_type == "raw":
        feat_dir = os.path.join("UCMerced_LandUse", "Images")
    elif args.feat_type == "pch" or args.feat_type == "phog" or args.feat_type == "cm":
        if args.feat_type == "pch" or args.feat_type == "phog":
            feat_dir = os.path.join("UCMerced_LandUse", "Features", args.feat_type, "level{:d}".format(args.level))
        else:
            feat_dir = os.path.join("UCMerced_LandUse", "Features", args.feat_type)

    if args.dist_metric == "SAD":
        val_dist = np.load(os.path.join(feat_dir, "val_SAD.npy"))
    elif args.dist_metric == "SSD":
        val_dist = np.load(os.path.join(feat_dir, "val_SSD.npy"))
    elif args.dist_metric == "NCC":
        val_dist = np.load(os.path.join(feat_dir, "val_NCC.npy"))

    precisions = np.zeros(n_classes)
    for i in range(n_classes):
        for j in range(val_size):
            neighbors = np.argsort(val_dist[i * val_size + j])[:args.num_retrieved]
            belongs = neighbors // train_size == i
            precisions[i] += belongs.sum()
        precisions[i] /= args.num_retrieved * val_size
        print("The precision for class {} is {:.3f}".format(classes[i], precisions[i]))

    print("The overall precision is {:.3f}".format(precisions.mean()))

if __name__ == '__main__':
    main()
