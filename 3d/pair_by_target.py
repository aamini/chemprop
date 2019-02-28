from argparse import ArgumentParser
from multiprocessing import Pool
import os
import sys
from tqdm import tqdm
from typing import Tuple

import numpy as np
from scipy.misc import comb

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import MoleculeDatapoint
from chemprop.data.utils import get_data


def target_overlap(data1: MoleculeDatapoint,
                   data2: MoleculeDatapoint) -> Tuple[float, int]:
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
        return -1, 0

    overlap = same / shared

    return overlap, shared


def compare_pair(data_pair: Tuple[MoleculeDatapoint, MoleculeDatapoint]) -> Tuple[str, str, Tuple[float, int]]:
    data1, data2 = data_pair
    smiles1, smiles2 = data1.smiles, data2.smiles
    overlap = target_overlap(data1, data2)

    return smiles1, smiles2, overlap


def pair_by_target(data_path: str,
                   save_path: str,
                   max_num_pairs: int):
    # Load data
    data = get_data(data_path)

    # Get pairs
    num_pairs = max_num_pairs or int(comb(len(data), 2))
    np.random.seed(0)
    data_pairs = np.random.choice(data.data, size=(num_pairs, 2))

    # Compute pairs
    with Pool() as pool:
        pairs = [pair for pair in tqdm(pool.imap_unordered(compare_pair, data_pairs), total=num_pairs)]

    import matplotlib.pyplot as plt
    from collections import Counter
    overlaps = [pair[-1][0] for pair in pairs]
    counts = Counter(overlaps)
    for key in sorted(counts.keys()):
        print(f'{key:.3f}: {counts[key]}')
    ones = [pair for pair in pairs if pair[-1][0] == 1.0]
    print(f'{len(ones)} / {num_pairs} ({len(ones) / num_pairs * 100}%) ones')
    plt.hist(overlaps, bins=100)
    plt.show()

    num_shared_ones = [pair[-1][1] for pair in ones]
    counts = Counter(num_shared_ones)
    for key in sorted(counts.keys()):
        print(f'{key}: {counts[key]}')

    plt.clf()
    plt.hist(num_shared_ones, bins=100)
    plt.show()
    # TODO: Save pairs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to classification dataset CSV')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to CSV where pairs data will be saved')
    parser.add_argument('--max_num_pairs', type=int, default=None,
                        help='Number of pairs to generate')
    args = parser.parse_args()

    pair_by_target(
        data_path=args.data_path,
        save_path=args.save_path,
        max_num_pairs=args.max_num_pairs
    )

"""
Chembl 1,000,000

-1.000: 892844
0.000: 55077
0.017: 2
0.018: 4
0.018: 1
0.018: 1
0.019: 4
0.019: 1
0.019: 3
0.020: 4
0.020: 4
0.021: 2
0.022: 5
0.023: 2
0.023: 3
0.024: 3
0.026: 2
0.027: 3
0.028: 1
0.029: 5
0.031: 3
0.032: 2
0.033: 2
0.034: 2
0.036: 1
0.037: 2
0.038: 1
0.039: 2
0.040: 1
0.042: 2
0.043: 2
0.045: 3
0.047: 1
0.048: 3
0.050: 1
0.053: 2
0.056: 1
0.057: 1
0.059: 2
0.060: 1
0.061: 1
0.062: 2
0.065: 2
0.067: 4
0.070: 1
0.071: 2
0.073: 1
0.077: 2
0.079: 2
0.083: 2
0.086: 1
0.091: 3
0.095: 1
0.100: 4
0.103: 1
0.105: 2
0.109: 1
0.111: 4
0.116: 1
0.125: 5
0.133: 1
0.135: 1
0.140: 1
0.143: 6
0.148: 1
0.149: 1
0.154: 2
0.167: 10
0.170: 1
0.172: 1
0.200: 31
0.204: 1
0.205: 1
0.214: 1
0.216: 1
0.222: 1
0.245: 1
0.250: 176
0.278: 1
0.280: 1
0.313: 1
0.316: 1
0.333: 849
0.353: 1
0.375: 1
0.381: 1
0.400: 87
0.414: 1
0.417: 1
0.429: 6
0.431: 1
0.444: 1
0.456: 1
0.458: 2
0.472: 1
0.500: 4922
0.553: 1
0.556: 2
0.558: 1
0.559: 1
0.571: 16
0.577: 1
0.600: 90
0.625: 8
0.630: 1
0.632: 1
0.667: 1397
0.706: 1
0.714: 11
0.727: 1
0.746: 1
0.750: 365
0.768: 1
0.778: 4
0.793: 1
0.800: 101
0.833: 31
0.844: 1
0.851: 1
0.857: 14
0.860: 1
0.860: 1
0.875: 7
0.889: 2
0.900: 2
0.949: 1
1.000: 43769
43769 / 1000000 (4.3769%) ones

Number of shared targets for overlap == 1.0
1: 34660
2: 7732
3: 1038
4: 236
5: 75
6: 15
7: 7
8: 4
9: 2
"""


"""
Chembl 1,000,000

-1.000: 892844
0.000: 55077
0.017: 2
0.018: 4
0.018: 1
0.018: 1
0.019: 4
0.019: 1
0.019: 3
0.020: 4
0.020: 4
0.021: 2
0.022: 5
0.023: 2
0.023: 3
0.024: 3
0.026: 2
0.027: 3
0.028: 1
0.029: 5
0.031: 3
0.032: 2
0.033: 2
0.034: 2
0.036: 1
0.037: 2
0.038: 1
0.039: 2
0.040: 1
0.042: 2
0.043: 2
0.045: 3
0.047: 1
0.048: 3
0.050: 1
0.053: 2
0.056: 1
0.057: 1
0.059: 2
0.060: 1
0.061: 1
0.062: 2
0.065: 2
0.067: 4
0.070: 1
0.071: 2
0.073: 1
0.077: 2
0.079: 2
0.083: 2
0.086: 1
0.091: 3
0.095: 1
0.100: 4
0.103: 1
0.105: 2
0.109: 1
0.111: 4
0.116: 1
0.125: 5
0.133: 1
0.135: 1
0.140: 1
0.143: 6
0.148: 1
0.149: 1
0.154: 2
0.167: 10
0.170: 1
0.172: 1
0.200: 31
0.204: 1
0.205: 1
0.214: 1
0.216: 1
0.222: 1
0.245: 1
0.250: 176
0.278: 1
0.280: 1
0.313: 1
0.316: 1
0.333: 849
0.353: 1
0.375: 1
0.381: 1
0.400: 87
0.414: 1
0.417: 1
0.429: 6
0.431: 1
0.444: 1
0.456: 1
0.458: 2
0.472: 1
0.500: 4922
0.553: 1
0.556: 2
0.558: 1
0.559: 1
0.571: 16
0.577: 1
0.600: 90
0.625: 8
0.630: 1
0.632: 1
0.667: 1397
0.706: 1
0.714: 11
0.727: 1
0.746: 1
0.750: 365
0.768: 1
0.778: 4
0.793: 1
0.800: 101
0.833: 31
0.844: 1
0.851: 1
0.857: 14
0.860: 1
0.860: 1
0.875: 7
0.889: 2
0.900: 2
0.949: 1
1.000: 43769
43769 / 1000000 (4.3769%) ones
"""
