from argparse import ArgumentParser
from multiprocessing import Pool
import os
import sys
from tqdm import tqdm
from typing import Optional, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import MoleculeDatapoint
from chemprop.data.utils import get_data


def target_overlap(data1: MoleculeDatapoint, data2: MoleculeDatapoint) -> float:
    target1, target2 = data1.targets, data2.targets
    same = shared = 0

    for t1, t2 in zip(target1, target2):
        if t1 is None or t2 is None:
            continue
            
        if t1 == 1 or t2 == 1:
            shared += 1

            if t1 == 1 and t2 == 1:
                same += 1

    if shared == 0:
        return -1

    overlap = same / shared

    return overlap


def determine_pair(data_pair: Tuple[MoleculeDatapoint, MoleculeDatapoint]) -> Tuple[str, str, float]:
    data1, data2 = data_pair
    smiles1, smiles2 = data1.smiles, data2.smiles
    overlap = target_overlap(data1, data2)
    # match = overlap == 1.0 if overlap is not None else None

    return smiles1, smiles2, overlap


def pair_by_target(data_path: str,
                   save_path: str,
                   num_pairs: int):
    # Load data
    data = get_data(data_path)

    # Get pairs
    data_pairs = (np.random.choice(data, size=2, replace=False) for _ in range(num_pairs))
    with Pool() as pool:
        pairs = [pair for pair in tqdm(pool.imap_unordered(determine_pair, data_pairs), total=num_pairs)]

    # import pdb; pdb.set_trace()

    import matplotlib.pyplot as plt
    overlaps = [pair[-1] for pair in pairs]
    plt.hist(overlaps, bins=100)
    plt.show()

    # TODO: Save pairs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to classification dataset CSV')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to CSV where pairs data will be saved')
    parser.add_argument('--num_pairs', type=int, default=1000,
                        help='Number of pairs to generate')
    args = parser.parse_args()

    pair_by_target(
        data_path=args.data_path,
        save_path=args.save_path,
        num_pairs=args.num_pairs
    )
