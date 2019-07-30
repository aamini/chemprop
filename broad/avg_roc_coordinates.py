from argparse import ArgumentParser
import csv
import os

import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve


def avg_roc(dirname: str, num_trials: int, num_folds: int):
    for j in range(num_trials):
        trial_rocs = []
        for i in range(num_folds):
            with open(os.path.join(dirname, f'trial_{j}', f'fold_{i}', 'pred.csv')) as f:
                pred = [float(row[1]) for row in csv.reader(f)]
            with open(os.path.join(dirname, f'trial_{j}', f'fold_{i}', 'true.csv')) as f:
                true = [int(row[1]) for row in csv.reader(f)]
            trial_rocs.append(roc_curve(true, pred))

        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        for roc in trial_rocs:
            fpr, tpr, _ = roc
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)

        with open(os.path.join(dirname, f'trial_{j}', 'roc.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['fpr,', 'tpr'])

            for fpr, tpr in zip(base_fpr, mean_tprs):
                writer.writerow(fpr, tpr)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='/data/rsg/chemistry/swansonk/chemprop_ckpt/213_finalens_six_trials',
                        help='Path to directory containing folders trial_0, trial_1, ...,'
                             'each of which contains files labelled fold_{i}_test_preds_sorted.csv')
    parser.add_argument('--num_trials', type=int, default=6,
                        help='Number of runs of cross-validation')
    parser.add_argument('--num_folds', type=int, default=20,
                        help='Number of cross-validation folds')
    args = parser.parse_args()

    avg_roc(
        dirname=args.dir,
        num_trials=args.num_trials,
        num_folds=args.num_folds
    )
