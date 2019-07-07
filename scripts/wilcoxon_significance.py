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
DATASETS['qm7'] = {'metric': mae}
DATASETS['qm8'] = {'metric': mae}
DATASETS['qm9'] = {'metric': mae}
DATASETS['delaney'] = {'metric': mse}
DATASETS['freesolv'] = {'metric': mse}
DATASETS['lipo'] = {'metric': mse}
DATASETS['pdbbind_full'] = {'metric': mse}
DATASETS['pdbbind_core'] = {'metric': mse}
DATASETS['pdbbind_refined'] = {'metric': mse}

# test if error of 1 is less than error of 2
COMPARISONS = [
    ('default', 'random_forest'),
    ('default', 'ffn_morgan'),
    ('default', 'ffn_morgan_counts'),
    ('default', 'ffn_rdkit'),
    ('features_no_opt', 'default'),
    ('hyperopt_eval', 'default'),
    ('hyperopt_eval', 'features_no_opt'),
    ('hyperopt_ensemble', 'default'),
    ('hyperopt_ensemble', 'hyperopt_eval'),
    ('default', 'undirected'),
    ('default', 'atom_messages'),
    # ('hyperopt_eval', 'mayr_et_al')
]


def load_preds_and_targets(preds_dir: str,
                           experiment: str,
                           dataset: str,
                           split_type: str) -> Tuple[np.ndarray, np.ndarray]:
    preds, targets = [], []
    for fold in range(10):
        preds_path = os.path.join(preds_dir, f'417_{experiment}', dataset, split_type, str(fold), 'preds.npy')
        targets_path = os.path.join(preds_dir, f'417_{experiment}', dataset, split_type, str(fold), 'targets.npy')

        if not (os.path.exists(preds_path) and os.path.exists(targets_path)):
            continue

        preds.append(np.load(preds_path))
        targets.append(np.load(targets_path))

    if len(preds) not in [3, 10]:
        raise ValueError('Did not find 3 or 10 preds/targets files')

    preds, targets = np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)

    return preds, targets


def wilcoxon_significance(preds_dir: str, split_type: str):
    print('\t'.join([f'{exp_1} vs {exp_2}' for exp_1, exp_2 in COMPARISONS]))

    for dataset in DATASETS:
        for exp_1, exp_2 in COMPARISONS:
            preds_1, targets_1 = load_preds_and_targets(preds_dir, exp_1, dataset, split_type)  # num_molecules x num_targets
            preds_2, targets_2 = load_preds_and_targets(preds_dir, exp_2, dataset, split_type)  # num_molecules x num_targets

            errors_1, errors_2 = DATASETS[dataset]['metric'](preds_1, targets_1), DATASETS[dataset]['metric'](preds_2, targets_2)

            # test if error of 1 is less than error of 2
            print(wilcoxon(errors_1, errors_2, alternative='less').pvalue, end='\t')
        print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--preds_dir', type=str, required=True,
                        help='Path to a directory containing predictions')
    parser.add_argument('--split_type', type=str, required=True, choices=['random', 'scaffold'],
                        help='Split type')
    args = parser.parse_args()

    wilcoxon_significance(
        preds_dir=args.preds_dir,
        split_type=args.split_type,
    )
