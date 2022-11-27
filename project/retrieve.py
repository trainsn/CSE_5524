import os
import argparse
import pdb
from collections import defaultdict
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieved-class', type=str, required=True, help='The class want to retrieve')
    parser.add_argument('--retrieved-idx', type=int, required=True, help='The retrieved image index in class')
    parser.add_argument('--num-retrieved', type=int, default=12, help='Number of images retrieved')
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
    valtest_size = val_size + test_size
    size_per_class = train_size + val_size + test_size

    class2idx = defaultdict(lambda: -1)
    for i in range(n_classes):
        class2idx[classes[i]] = i

    rnd_idx = np.load("rnd_idx.npy")
    ori2permed = np.zeros(size_per_class, dtype=int)
    ori2permed[rnd_idx] = np.arange(size_per_class)

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
        dist = np.load(os.path.join(feat_dir, "NCC.npy"))

    retrieved_class_idx = class2idx[args.retrieved_class]
    if retrieved_class_idx == -1:
        print("The class you input is not in the database!")
        return

    retrieved_permed_idx = ori2permed[args.retrieved_idx] - train_size
    if retrieved_permed_idx < 0:
        print("You are retrieving a template image!")
        return

    neighbors = np.argsort(dist[retrieved_class_idx * valtest_size + retrieved_permed_idx])[:args.num_retrieved]
    neighbor_class_idx = neighbors // train_size
    neighbor_inclass_idx = rnd_idx[neighbors % train_size]
    for i in range(args.num_retrieved):
        print("The rank {:d} most similar image is {}{:02d}".format(i + 1, classes[neighbor_class_idx[i]], neighbor_inclass_idx[i]))

if __name__ == '__main__':
    main()
