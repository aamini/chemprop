from argparse import ArgumentParser
from collections import OrderedDict
import csv
import os
from typing import Tuple

import numpy as np
from scipy.stats import wilcoxon


def mae(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(preds - targets), axis=1)


def mse(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.mean((preds - targets) ** 2, axis=1)


DATASETS = OrderedDict()
DATASETS['qm7'] = {'metric': mae, 'folds': list(range(10))}
DATASETS['qm8'] = {'metric': mae, 'folds': list(range(10))}
DATASETS['qm9'] = {'metric': mae, 'folds': list(range(3))}
DATASETS['delaney'] = {'metric': mse, 'folds': list(range(10))}
DATASETS['freesolv'] = {'metric': mse, 'folds': list(range(10))}
DATASETS['lipo'] = {'metric': mse, 'folds': list(range(10))}
DATASETS['pdbbind_full'] = {'metric': mse, 'folds': list(range(10))}
DATASETS['pdbbind_core'] = {'metric': mse, 'folds': list(range(10))}
DATASETS['pdbbind_refined'] = {'metric': mse, 'folds': list(range(10))}

# test if error of 1 is less than error of 2
COMPARISONS = [
    ('default', 'ffn_morgan')
]


def load_preds_and_targets(preds_dir: str,
                           experiment: str,
                           dataset: str,
                           split_type: str) -> Tuple[np.ndarray, np.ndarray]:
    all_preds, all_targets = [], []
    for fold in DATASETS[dataset]['folds']:
        preds_path = os.path.join(preds_dir, f'417_{experiment}', dataset, split_type, str(fold), 'preds.npy')
        preds = np.load(preds_path)
        all_preds.append(preds)

        targets_path = os.path.join(preds_dir, f'417_{experiment}', dataset, split_type, str(fold), 'targets.npy')
        targets = np.load(targets_path)
        all_targets.append(targets)

    all_preds, all_targets = np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)

    return all_preds, all_targets


def wilcoxon_significance(preds_dir: str, split_type: str):
    for dataset in DATASETS:
        for exp_1, exp_2 in COMPARISONS:
            preds_1, targets_1 = load_preds_and_targets(preds_dir, exp_1, dataset, split_type)  # num_molecules x num_targets
            preds_2, targets_2 = load_preds_and_targets(preds_dir, exp_2, dataset, split_type)  # num_molecules x num_targets

            errors_1, errors_2 = DATASETS[dataset]['metric'](preds_1, targets_1), DATASETS[dataset]['metric'](preds_2, targets_2)

            # test if error of 1 is less than error of 2
            print(wilcoxon(errors_1, errors_2, alternative='less').pvalue, end='\t')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--preds_dir', type=str, required=True,
                        help='Path to a directory containing predictions')
    parser.add_argument('--split_type', type=str, required=True,
                        help='Split type')
    args = parser.parse_args()

    wilcoxon_significance(
        preds_dir=args.preds_dir,
        split_type=args.split_type,
    )
