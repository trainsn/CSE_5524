import os
import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5, help='parameter k for KNN')
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

    confusion = np.zeros((n_classes, n_classes), dtype=int)
    acc = 0
    for i in range(n_classes):
        for j in range(valtest_size):
            neighbors = np.argsort(dist[i * valtest_size + j])[:args.k]
            neighbor_class_idx = neighbors // train_size
            pred = stats.mode(neighbor_class_idx)[0][0]
            acc += pred == i
            confusion[pred][i] += 1
    acc /= n_classes * valtest_size
    print("Overall accuracy: {:.3f}".format(acc))

    for i in range(n_classes):
        binary_conf = np.zeros((2, 2), dtype=int)
        not_i = np.ones(n_classes, dtype=bool)
        not_i[i] = False
        binary_conf[0, 0] = confusion[i][i]
        binary_conf[0, 1] = confusion[i][not_i].sum()
        binary_conf[1, 0] = confusion[not_i][:, i].sum()
        binary_conf[1, 1] = confusion[not_i][:, not_i].sum()

        precision = binary_conf[0, 0] / (binary_conf[0, 0] + binary_conf[0, 1])
        recall = binary_conf[0, 0] / (binary_conf[0, 0] + binary_conf[1, 0])
        f1 = 2 * precision * recall / (precision + recall)
        print("Class {},\tprecision: {:.3f},\trecall: {:.3f},\tF1: {:.3f}".format(classes[i], precision, recall, f1))

    fig, ax = plt.subplots(figsize=(13, 13))
    ax.matshow(confusion, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(x=j, y=i, s=confusion[i, j], va='center', ha='center', size='medium')
    plt.xlabel('Actuals', fontsize=18)
    plt.ylabel('Predictions', fontsize=18)
    x_major_locator=MultipleLocator(1)
    y_major_locator=MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xticklabels(['']+classes, rotation=90, fontsize=8)
    ax.set_yticklabels(['']+classes, fontsize=8)
    plt.show()


if __name__ == '__main__':
    main()
