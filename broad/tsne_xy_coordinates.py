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


def get_smiles(path: str) -> List[str]:
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        smiles = [line[0] for line in reader if len(line) > 0]

    return smiles


def tsne_xy_coordinates(smiles_paths: List[str], save_dir: str):
    assert len(smiles_paths) <= len(COLORS)

    # Load the smiles datasets
    print('Loading data')
    smiles, colors, scopes = [], [], []
    for smiles_path, color in zip(smiles_paths, COLORS):
        new_smiles = get_smiles(smiles_path)
        print(f'{os.path.basename(smiles_path)}: {len(new_smiles):,}')
        scopes += slice(len(smiles), len(new_smiles))
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

    # Save xy coordinates
    for smiles_path, scope in zip(smiles_paths, scopes):
        xy_path = os.path.join(save_dir, os.path.basename(smiles_path).replace('.csv', '_xy.csv'))

        with open(xy_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles', 'x', 'y'])

            for smile, (x, y) in zip(smiles[scope], X[scope]):
                writer.writerow([smile, x, y])

    # Generate t-SNE plot
    plt.clf()
    plt.figure(figsize=(6.4 * 10, 4.8 * 10))

    for (x, y), color in zip(X, colors):
        plt.plot(x[0], x[1], marker='o', markersize=20, color=color)

    plt.xticks([]), plt.yticks([])
    plt.savefig(os.path.join(save_dir, 'tsne.png'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--smiles_paths', nargs='+', type=str, required=True,
                        help='Path to .csv files containing smiles strings (with header)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to a directory where the t-SNE xy coordinates and plot will be saved')
    args = parser.parse_args()

    tsne_xy_coordinates(
        smiles_paths=args.smiles_paths,
        save_dir=args.save_dir
    )
