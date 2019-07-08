from argparse import ArgumentParser
from collections import OrderedDict
import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon
from tqdm import tqdm

from chemprop.train.evaluate import evaluate_predictions
from chemprop.utils import mean_absolute_error, rmse, roc_auc_score, prc_auc


DATASETS = OrderedDict()
DATASETS['qm7'] = {'metric': mean_absolute_error, 'type': 'regression'}
DATASETS['qm8'] = {'metric': mean_absolute_error, 'type': 'regression'}
DATASETS['qm9'] = {'metric': mean_absolute_error, 'type': 'regression'}
DATASETS['delaney'] = {'metric': rmse, 'type': 'regression'}
DATASETS['freesolv'] = {'metric': rmse, 'type': 'regression'}
DATASETS['lipo'] = {'metric': rmse, 'type': 'regression'}
DATASETS['pdbbind_full'] = {'metric': rmse, 'type': 'regression'}
DATASETS['pdbbind_core'] = {'metric': rmse, 'type': 'regression'}
DATASETS['pdbbind_refined'] = {'metric': rmse, 'type': 'regression'}
DATASETS['pcba'] = {'metric': prc_auc, 'type': 'classification'}
DATASETS['muv'] = {'metric': prc_auc, 'type': 'classification'}
DATASETS['hiv'] = {'metric': roc_auc_score, 'type': 'classification'}
DATASETS['bace'] = {'metric': roc_auc_score, 'type': 'classification'}
DATASETS['bbbp'] = {'metric': roc_auc_score, 'type': 'classification'}
DATASETS['tox21'] = {'metric': roc_auc_score, 'type': 'classification'}
DATASETS['toxcast'] = {'metric': roc_auc_score, 'type': 'classification'}
DATASETS['sider'] = {'metric': roc_auc_score, 'type': 'classification'}
DATASETS['clintox'] = {'metric': roc_auc_score, 'type': 'classification'}
DATASETS['chembl'] = {'metric': roc_auc_score, 'type': 'classification'}

# test if 1 is better than 2 (less error, higher auc)
COMPARISONS = [
    ('default', 'random_forest'),
    ('default', 'ffn_morgan'),
    ('default', 'ffn_morgan_count'),
    ('default', 'ffn_rdkit'),
    ('features_no_opt', 'default'),
    ('hyperopt_eval', 'default'),
    ('hyperopt_ensemble', 'default'),
    ('hyperopt_eval', 'features_no_opt'),
    ('hyperopt_ensemble', 'hyperopt_eval'),
    ('default', 'undirected'),
    ('default', 'atom_messages'),
    ('hyperopt_eval', 'compare_lsc_scaffold')
]


def load_preds_and_targets(preds_dir: str,
                           experiment: str,
                           dataset: str,
                           split_type: str) -> Tuple[Optional[List[List[float]]],
                                                     Optional[List[List[float]]]]:
    all_preds, all_targets = [], []
    num_folds = 0
    for fold in range(10):
        preds_path = os.path.join(preds_dir, f'417_{experiment}', dataset, split_type, str(fold), 'preds.npy')
        targets_path = os.path.join(preds_dir, f'417_{experiment}', dataset, split_type, str(fold), 'targets.npy')

        if not (os.path.exists(preds_path) and os.path.exists(targets_path)):
            continue

        preds = np.load(preds_path)
        targets = np.load(targets_path)

        preds = [[p if not np.isnan(p) else None for p in pred] for pred in preds]
        targets = [[t if not np.isnan(t) else None for t in target] for target in targets]

        all_preds += preds
        all_targets += targets

        num_folds += 1

    if num_folds not in [3, 10]:
        # print(f'Did not find 3 or 10 preds/targets files for experiment "{experiment}" and dataset "{dataset}" and split type "{split_type}"')
        return None, None

    return all_preds, all_targets


def compute_values(dataset: str,
                   preds: List[List[List[float]]],
                   targets: List[List[List[float]]]) -> List[float]:
    num_tasks = len(preds[0][0])

    values = [
        evaluate_predictions(
            preds=pred,
            targets=target,
            num_tasks=num_tasks,
            metric_func=DATASETS[dataset]['metric'],
            dataset_type=DATASETS[dataset]['type']
        )
        for pred, target in tqdm(zip(preds, targets), total=len(preds))
    ]

    values = [np.nanmean(value) for value in values]

    return values


def wilcoxon_significance(preds_dir: str, split_type: str):
    print('dataset\t' + '\t'.join([f'{exp_1} vs {exp_2}' for exp_1, exp_2 in COMPARISONS]))

    for dataset in DATASETS:
        print(dataset, end='\t')
        dataset_type = DATASETS[dataset]['type']

        for exp_1, exp_2 in COMPARISONS:
            if exp_2 == 'compare_lsc_scaffold' and split_type != 'scaffold':
                continue

            preds_1, targets_1 = load_preds_and_targets(preds_dir, exp_1, dataset, split_type)  # num_molecules x num_targets
            preds_2, targets_2 = load_preds_and_targets(preds_dir, exp_2, dataset, split_type)  # num_molecules x num_targets

            if any(x is None for x in [preds_1, targets_1, preds_2, targets_2]):
                print('Error', end='\t')
                continue

            if dataset_type == 'regression':
                preds_1, targets_1 = [[pred] for pred in preds_1], [[target] for target in targets_1]
                preds_2, targets_2 = [[pred] for pred in preds_2], [[target] for target in targets_2]
            else:
                # Split into 30 roughly equal pieces
                preds_1, targets_1 = np.array_split(preds_1, 30), np.array_split(targets_1, 30)
                preds_2, targets_2 = np.array_split(preds_2, 30), np.array_split(targets_2, 30)

            values_1, values_2 = compute_values(dataset, preds_1, targets_1), compute_values(dataset, preds_2, targets_2)

            # test if error of 1 is less than error of 2
            print(wilcoxon(values_1, values_2, alternative='less' if dataset_type == 'regression' else 'greater').pvalue, end='\t')
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
