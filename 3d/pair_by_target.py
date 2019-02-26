from argparse import ArgumentParser
from itertools import combinations
from multiprocessing import Pool
import os
import sys
from tqdm import tqdm
from typing import Generator, Tuple

from scipy.misc import comb

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data.utils import get_data


def target_overlap(data1: MoleculeDatapoint,
                   data2: MoleculeDatapoint) -> float:
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


def pairs_generator(data: MoleculeDataset,
                    num_pairs: int) -> Generator[Tuple[MoleculeDatapoint, MoleculeDatapoint], None, None]:
    for i, (data1, data2) in enumerate(combinations(data, 2)):
        if i >= num_pairs:
            return
        yield data1, data2


def pair_by_target(data_path: str,
                   save_path: str,
                   max_num_pairs: int):
    # Load data
    data = get_data(data_path)

    # Shuffle
    # TODO: instead do randomness in the pairs generator or something
    data.shuffle(seed=0)

    # Get pairs
    num_pairs = max_num_pairs or int(comb(len(data), 2))
    data_pairs = pairs_generator(data, num_pairs)
    with Pool() as pool:
        pairs = [pair for pair in tqdm(pool.imap_unordered(determine_pair, data_pairs), total=num_pairs)]

    # import pdb; pdb.set_trace()

    import matplotlib.pyplot as plt
    from collections import Counter
    overlaps = [pair[-1] for pair in pairs]
    plt.hist(overlaps, bins=100)
    plt.show()

    counts = Counter(overlaps)

    import pdb; pdb.set_trace()

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
