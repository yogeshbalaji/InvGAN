# Code for plotting accuracy vs reconstruction error

import os
import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse


# TODO: need to put this in utils
def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_file', required=True, help='path to source dataset')
    opt = parser.parse_args()

    # Creating folders
    results_fname = "".join(opt.pkl_file.split('/')[-1].split('=')[:-1]) + '_plots'
    results_path = os.path.join(os.path.dirname(opt.pkl_file), results_fname)
    mkdirp(results_path)

    # Curve params
    num_disc = 100 # Number of discretizations

    with open(opt.pkl_file) as f:
        roc_info = pickle.load(f)

    [all_labels, preds, diffs, z_norms] = roc_info

    min_diff = np.min(diffs)
    max_diff = np.max(diffs)

    x_arr = []
    y_arr = []

    # Plot 1 -- acc vs recons
    for i, level in enumerate(np.linspace(min_diff, max_diff, num_disc)):

        if i == 0:
            continue
        indices = np.where(diffs<=level)[0]
        correct = np.sum(preds[indices] == all_labels[indices])
        total = len(indices)
        acc = float(correct)/float(total)
        x_arr.append(level)
        y_arr.append(acc)

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    plt.xlabel('Reconstruction threshold')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.plot(x_arr, y_arr)
    plt.savefig('{}/rec_acc.png'.format(results_path))

    # Plot 2: acc v z_norm
    plt.clf()
    min_norm = np.min(z_norms)
    max_norm = np.max(z_norms)

    x_arr = []
    y_arr = []

    for i, level in enumerate(np.linspace(min_norm, max_norm, num_disc)):
        if i == 0:
            continue
        indices = np.where(z_norms <= level)[0]
        correct = np.sum(preds[indices] == all_labels[indices])
        total = len(indices)
        acc = float(correct) / float(total)
        x_arr.append(level)
        y_arr.append(acc)

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    plt.xlabel('Norm z threshold')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.plot(x_arr, y_arr)
    plt.savefig('{}/norm_acc.png'.format(results_path))


if __name__ == '__main__':
    main()
