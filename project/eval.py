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
    parser.add_argument('--dataset', type=str, help='val or test')
    args = parser.parse_args()
    print(args)

    classes = ["agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral", "denseresidential",
               "forest", "freeway", "golfcourse", "harbor", "intersection", "mediumresidential", "mobilehomepark",
               "overpass", "parkinglot", "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"]

    n_classes = len(classes)
    train_size, val_size, test_size = 80, 10, 10

    if args.feat_type == "raw":
        feat_dir = os.path.join("UCMerced_LandUse", "Images")
    elif args.feat_type == "pch" or args.feat_type == "phog" or args.feat_type == "cm":
        if args.feat_type == "pch" or args.feat_type == "phog":
            feat_dir = os.path.join("UCMerced_LandUse", "Features", args.feat_type, "level{:d}".format(args.level))
        else:
            feat_dir = os.path.join("UCMerced_LandUse", "Features", args.feat_type)

    if args.dist_metric == "SAD":
        dist = np.load(os.path.join(feat_dir, "SAD.npy"))
    elif args.dist_metric == "SSD":
        dist = np.load(os.path.join(feat_dir, "SSD.npy"))
    elif args.dist_metric == "NCC":
        dist = -np.load(os.path.join(feat_dir, "NCC.npy"))

    candi = []
    if args.dataset == "val":
        for i in range(n_classes):
            candi.extend(np.arange(i * (val_size + test_size), i * (val_size + test_size) + val_size))
        valtest_size = val_size
    elif args.dataset == "test":
        for i in range(n_classes):
            candi.extend(np.arange(i * (val_size + test_size) + val_size, (i + 1) * (val_size + test_size)))
        valtest_size = test_size
    candi = np.array(candi)
    dist = dist[candi]

    precisions = np.zeros(n_classes)
    for i in range(n_classes):
        for j in range(valtest_size):
            neighbors = np.argsort(dist[i * valtest_size + j])[:args.num_retrieved]
            belongs = neighbors // train_size == i
            precisions[i] += belongs.sum()
        precisions[i] /= args.num_retrieved * valtest_size
        print("The accuracy for class {} is {:.3f}".format(classes[i], precisions[i]))

    print("The overall accuracy is {:.3f}".format(precisions.mean()))

if __name__ == '__main__':
    main()
