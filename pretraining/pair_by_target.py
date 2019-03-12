from argparse import ArgumentParser
import csv
import os
import sys
from typing import List, Tuple

import numpy as np
from scipy.misc import comb
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, split_data
from chemprop.utils import makedirs


def create_pairs(data: MoleculeDataset,
                 num_positive_pairs: int,
                 num_negative_pairs: int) -> Tuple[List[Tuple[str, str]],
                                                   List[Tuple[str, str]]]:
    # Get smiles
    smiles = data.smiles()

    # Get targets
    targets = np.array(data.targets()).T  # (num_tasks, num_molecules)
    ones = [np.where(target == 1)[0] for target in targets]
    zeros = [np.where(target == 0)[0] for target in targets]

    # Determine which targets to sample
    positive_pair_counts = np.array([comb(len(one), 2) for one in ones])
    positive_pair_weights = positive_pair_counts / sum(positive_pair_counts)
    negative_pair_counts = np.array([len(zero) * len(one) for zero, one in zip(zeros, ones)])
    negative_pair_weights = negative_pair_counts / sum(negative_pair_counts)

    positive_pair_targets = np.random.choice(len(targets), size=num_positive_pairs, p=negative_pair_weights)
    negative_pair_targets = np.random.choice(len(targets), size=num_negative_pairs, p=positive_pair_weights)

    # Sample pairs
    positive_pairs = []
    for target in tqdm(positive_pair_targets, total=num_positive_pairs):
        if len(ones[target]) < 2:
            continue
        i, j = np.random.choice(ones[target], size=2, replace=False)
        positive_pairs.append((smiles[i], smiles[j]))

    negative_pairs = []
    for target in tqdm(negative_pair_targets, total=num_negative_pairs):
        if len(zeros[target]) == 0 or len(ones[target]) == 0:
            continue
        i = np.random.choice(zeros[target])
        j = np.random.choice(ones[target])
        negative_pairs.append((smiles[i], smiles[j]))

    return positive_pairs, negative_pairs


def pair_by_target(data_path: str,
                   save_dir: str,
                   num_train_pairs: int,
                   num_val_pairs: int,
                   num_test_pairs: int,
                   percent_positive: float):
    # Load data
    data = get_data(data_path)

    # Split data
    train_data, val_data, test_data = split_data(data=data)

    # Create pairs
    train_pairs = create_pairs(
        data=train_data,
        num_positive_pairs=int(num_train_pairs * percent_positive),
        num_negative_pairs=int(num_train_pairs * (1 - percent_positive))
    )
    val_pairs = create_pairs(
        data=val_data,
        num_positive_pairs=int(num_val_pairs * percent_positive),
        num_negative_pairs=int(num_val_pairs * (1 - percent_positive))
    )
    test_pairs = create_pairs(
        data=test_data,
        num_positive_pairs=int(num_test_pairs * percent_positive),
        num_negative_pairs=int(num_test_pairs * (1 - percent_positive))
    )

    # Save pairs
    makedirs(save_dir)
    for split, pairs in zip(['train', 'val', 'test'], [train_pairs, val_pairs, test_pairs]):
        positive_pairs, negative_pairs = pairs
        with open(os.path.join(save_dir, f'{split}.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles1', 'smiles2', 'pair'])
            for smiles1, smiles2 in positive_pairs:
                writer.writerow([smiles1, smiles2, 1])
            for smiles1, smiles2 in negative_pairs:
                writer.writerow([smiles1, smiles2, 0])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to classification dataset CSV')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to directory where CSVs with pairs will be saved')
    parser.add_argument('--num_train_pairs', type=int, required=True,
                        help='Number of train pairs')
    parser.add_argument('--num_val_pairs', type=int, required=True,
                        help='Number of train pairs')
    parser.add_argument('--num_test_pairs', type=int, required=True,
                        help='Number of train pairs')
    parser.add_argument('--percent_positive', type=float, default=0.5,
                        help='Percent of pairs which are positive')
    args = parser.parse_args()

    pair_by_target(
        data_path=args.data_path,
        save_dir=args.save_dir,
        num_train_pairs=args.num_train_pairs,
        num_val_pairs=args.num_val_pairs,
        num_test_pairs=args.num_test_pairs,
        percent_positive=args.percent_positive
    )
