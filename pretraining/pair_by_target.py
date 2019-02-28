from argparse import ArgumentParser
import csv
import os
import sys

import numpy as np
from scipy.misc import comb
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_data
from chemprop.utils import makedirs


def pair_by_target(data_path: str,
                   save_path: str,
                   num_positive_pairs: int,
                   num_negative_pairs: int):
    # Load data
    data = get_data(data_path)
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
        i, j = np.random.choice(ones[target], size=2, replace=False)
        positive_pairs.append((smiles[i], smiles[j]))

    negative_pairs = []
    for target in tqdm(negative_pair_targets, total=num_negative_pairs):
        i = np.random.choice(zeros[target])
        j = np.random.choice(ones[target])
        negative_pairs.append((smiles[i], smiles[j]))

    # Save pairs
    makedirs(save_path, isfile=True)
    with open(save_path, 'w') as f:
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
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to CSV where pairs data will be saved')
    parser.add_argument('--num_positive_pairs', type=int, required=True,
                        help='Number of positive pairs to generate')
    parser.add_argument('--num_negative_pairs', type=int, required=True,
                        help='Number of negative pairs to generate')
    args = parser.parse_args()

    pair_by_target(
        data_path=args.data_path,
        save_path=args.save_path,
        num_positive_pairs=args.num_positive_pairs,
        num_negative_pairs=args.num_negative_pairs
    )
