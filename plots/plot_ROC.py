# Code for plotting accuracy vs reconstruction error

import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import argparse
from sklearn import metrics


# TODO: need to put this in utils
def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_file_adv', required=True, help='path to source dataset')
    parser.add_argument('--pkl_file_clean', required=True, help='path to source dataset')
    opt = parser.parse_args()

    # Creating folders
    results_fname = "".join(opt.pkl_file_adv.split('/')[-1].split('.pkl')[0].split('=')) + '_plots'
    results_path = os.path.join(os.path.dirname(opt.pkl_file_adv), results_fname)
    mkdirp(results_path)

    with open(opt.pkl_file_adv) as f:
        roc_info_adv = pickle.load(f)

    with open(opt.pkl_file_clean) as f:
        roc_info_clean = pickle.load(f)

    [all_labels_adv, preds_adv, diffs_adv, z_norms_adv] = roc_info_adv
    [all_labels_clean, preds_clean, diffs_clean, z_norms_clean] = roc_info_clean
    
    # Plotting a histogram of diffs
    
    plt.figure()
    plt.hist(diffs_adv, bins=100, color='red', alpha=0.3)
    plt.hist(diffs_clean, bins=100, color='green', alpha=0.3)
    plt.savefig('{}/err_hist.png'.format(results_path))

    plt.figure()
    plt.hist(z_norms_adv, bins=100, color='red', alpha=0.3)
    plt.hist(z_norms_clean, bins=100, color='green', alpha=0.3)
    plt.savefig('{}/z_hist.png'.format(results_path))

    plt.clf()
    # ROC curve
    diffs_all = np.concatenate((diffs_clean, diffs_adv))
    diffs_all = (diffs_all - np.min(diffs_all))/(np.max(diffs_all) - np.min(diffs_all))
    cls_labels_all = np.concatenate((np.zeros(diffs_clean.shape), np.ones(diffs_adv.shape)))
    
    y_true = cls_labels_all
    y_pred = diffs_all
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.roc_auc_score(y_true, y_pred)
    
    plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('{}/ROC.png'.format(results_path))
    print('AUC: {}'.format(roc_auc))
    

if __name__ == '__main__':
    main()
