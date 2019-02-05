from argparse import ArgumentParser
import csv

import numpy as np
from sklearn.metrics import precision_score


def precision(dirname: str, true_type: str, count: int, total: int, num_folds: int):
    true_colname = f'50uM_{true_type}_Avg_bin'
    percent = count / total
    scores = []
    for i in range(num_folds):
        with open(f'{dirname}/fold_{i}_test_preds_sorted.csv') as f:
            data = list(csv.DictReader(f))
        true = [int(row[true_colname]) for row in data]
        num_pred_hits = int(percent * len(data))
        preds = [1] * num_pred_hits + [0] * (len(data) - num_pred_hits)
        precision = precision_score(true, preds)
        scores.append(precision)
    print(f'precision = {np.mean(scores)} +/- {np.std(scores)}')
    print(f'{np.mean(scores) * count} hits out of {count} predictions')

    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='19.01.30_50uMInhibition_classification_x6',
                        help='Path to directory containing folders trial_0, trial_1, ...,'
                             'each of which contains files labelled fold_{i}_test_preds_sorted.csv')
    parser.add_argument('--true_type', type=str, default='killing',
                        choices=['Inhibition', 'killing'],
                        help='The type of a true hit')
    parser.add_argument('--count', type=int, default=100,
                        help='Number of molecules to predict out of args.total')
    parser.add_argument('--total', type=int, default=10000,
                        help='Number of molecules in held out evaluation set, of which args.count will be predicted')
    parser.add_argument('--num_folds', type=int, default=20,
                        help='Number of cross-validation folds')
    args = parser.parse_args()

    precision(
        dirname=args.dir,
        true_type=args.true_type,
        count=args.count,
        total=args.total,
        num_folds=args.num_folds
    )
