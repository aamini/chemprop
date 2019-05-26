from argparse import ArgumentParser
import csv
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.features import get_features_generator
from chemprop.utils import makedirs


COLORS = ['blue', 'red', 'green', 'cyan']
su3327_smiles = 'Nc1nnc(Sc2ncc(s2)[N+]([O-])=O)s1'


def get_smiles(path: str) -> List[str]:
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        smiles = [line[0] for line in reader if len(line) > 0]

    return smiles


def compare_datasets_tsne(smiles_paths: List[str], highlight_paths: List[str], save_dir: str):
    assert len(smiles_paths) <= len(COLORS)

    # Load the smiles datasets
    print('Loading data')
    smiles, colors = [], []
    for smiles_path, color in zip(smiles_paths, COLORS):
        new_smiles = get_smiles(smiles_path)
        print(f'{os.path.basename(smiles_path)}: {len(new_smiles):,}')
        smiles += new_smiles
        colors += [color] * len(new_smiles)

    # Compute Morgan fingerprints
    print('Computing Morgan fingerprints')
    morgan_generator = get_features_generator('morgan')
    morgans = [morgan_generator(smile) for smile in tqdm(smiles, total=len(smiles))]

    print('Running t-SNE')
    import time
    start = time.time()
    tsne = TSNE(n_components=2, init='pca', random_state=0, metric='jaccard')
    X = tsne.fit_transform(morgans)
    print(f'time = {time.time() - start}')

    print('Plotting t-SNE')
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    makedirs(save_dir)

    # Ensure plotting a version without highlighting
    if None not in highlight_paths:
        highlight_paths = [None] + highlight_paths

    # One plot for each highlight path
    for highlight_path in highlight_paths:
        if highlight_path is not None:
            highlight_name = os.path.splitext(os.path.basename(highlight_path))[0]
            save_path = os.path.join(save_dir, f'{highlight_name}_highlight.pdf')
            highlight_smiles = set(get_smiles(highlight_path))
        else:
            save_path = os.path.join(save_dir, 'no_highlight.pdf')
            highlight_smiles = set()

        print(os.path.basename(save_path))

        plt.clf()
        plt.figure(figsize=(6.4 * 10, 4.8 * 10))

        for smile, x, color in zip(smiles, X, colors):
            if smile not in highlight_smiles:
                plt.plot(x[0], x[1], marker='o', markersize=20, color=color)
        for smile, x, color in zip(smiles, X, colors):
            if smile in highlight_smiles:
                plt.plot(x[0], x[1], marker='*', markersize=40, color='gold')

        plt.xticks([]), plt.yticks([])

        plt.savefig(os.path.join(save_path))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--smiles_paths', nargs='+', type=str, required=True,
                        help='Path to .csv files containing smiles strings (with header)')
    parser.add_argument('--highlight_paths', nargs='+', type=str, default=[None],
                        help='Path to .csv files that should be highlighted in the plot')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to a directory where the t-SNE plots will be saved')
    args = parser.parse_args()

    compare_datasets_tsne(
        smiles_paths=args.smiles_paths,
        highlight_paths=args.highlight_paths,
        save_dir=args.save_dir
    )
