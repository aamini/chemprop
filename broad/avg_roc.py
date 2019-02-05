from argparse import ArgumentParser
import csv
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_auc_score, roc_curve


def score(data: List[Dict[str, str]], true_colname: str, preds_colname: str):
    true = [int(row[true_colname]) for row in data]
    preds = [float(row[preds_colname]) for row in data]
    ranks = np.where(true)[0] + 1
    if len(ranks) == 0:
        return None

    return np.mean(ranks), np.median(ranks), roc_auc_score(true, preds), roc_curve(true, preds)


def avg_roc(dirname: str, true_type: str, pred_type: str, num_trials: int, num_folds: int):
    true_colname = f'50uM_{true_type}_Avg_bin'
    preds_colname = f'50uM_{pred_type}_pred'
    all_tprs = []
    for j in range(num_trials):
        scores = []
        for i in range(num_folds):
            with open(f'{dirname}/trial_{j}/fold_{i}_test_preds_sorted.csv') as f:
                data = list(csv.DictReader(f))
            score_ = score(data, true_colname, preds_colname)
            if score_ is not None:
                scores.append(score_)
        _, _, _, trial_rocs = zip(*scores)

        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        for roc in trial_rocs:
            fpr, tpr, _ = roc
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        all_tprs.append(mean_tprs)

    plt.clf()
    base_fpr = np.linspace(0, 1, 101)
    for tpr in all_tprs:
        plt.plot(base_fpr, tpr, 'b', alpha=0.15)

    all_tprs = np.array(all_tprs)
    mean_tprs = all_tprs.mean(axis=0)
    std = all_tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, 'b')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.title(f'ROC for {pred_type} model ranking {true_type}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig(f'{dirname}/roc_ranking_{true_type}_with_{pred_type}_model_avg.png')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='19.01.30_50uMInhibition_classification_x6',
                        help='Path to directory containing folders trial_0, trial_1, ...,'
                             'each of which contains files labelled fold_{i}_test_preds_sorted.csv')
    parser.add_argument('--true_type', type=str, default='killing',
                        choices=['Inhibition', 'killing'],
                        help='The type of a true hit')
    parser.add_argument('--pred_type', type=str, default='Inhibition',
                        choices=['Inhibition', 'killing'],
                        help='The type that the model predicts')
    parser.add_argument('--num_trials', type=int, default=6,
                        help='Number of runs of cross-validation')
    parser.add_argument('--num_folds', type=int, default=20,
                        help='Number of cross-validation folds')
    args = parser.parse_args()

    avg_roc(
        dirname=args.dir,
        true_type=args.true_type,
        pred_type=args.pred_type,
        num_trials=args.num_trials,
        num_folds=args.num_folds
    )
