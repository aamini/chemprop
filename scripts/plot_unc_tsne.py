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
        smiles = []
        unc = []
        for line in reader:
            if len(line) > 0:
                smiles.append(line[0])
                # HARDCODED: this has to be uncertainty
                if len(line) > 1:
                    unc.append(float(line[1]))
                else:
                    unc.append(None)
    return smiles, unc

def tsne_xy_coordinates(smiles_paths: List[str], save_dir: str):
    assert len(smiles_paths) <= len(COLORS)

    # Load the smiles datasets
    print('Loading data')
    smiles, colors, scopes, unc_coloring = [], [], [], []
    for smiles_path, color in zip(smiles_paths, COLORS):
        new_smiles, new_unc = get_smiles(smiles_path)
        print(f'{os.path.basename(smiles_path)}: {len(new_smiles):,}')
        scopes.append(slice(len(smiles), len(smiles) + len(new_smiles)))
        smiles += new_smiles

        if new_unc[0] != None:
            unc_coloring += np.log(new_unc).tolist()

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
    plt.close()
    plt.figure(figsize=[6.4 * 2, 4.8 * 2])

    x0 = np.split(X[scopes[0]],2,1)[0]
    y0 = np.split(X[scopes[0]],2,1)[1]

    x1 = np.split(X[scopes[1]],2,1)[0]
    y1 = np.split(X[scopes[1]],2,1)[1]

    # Color Broad data by uncertainties; primary training orange
    plt.scatter(x1, y1, c=unc_coloring, s=6, alpha=1, zorder=2, cmap='Blues')
    plt.colorbar()
    plt.scatter(x0, y0, c='orange', s=10, alpha=1, zorder=1)

    plt.xticks([]), plt.yticks([])
    plt.savefig(os.path.join(save_dir, 'unc_tsne.pdf'))
    plt.show()
    plt.close()

if __name__ == '__main__':
    parser = ArgumentParser()

    ## First arg: path to Stokes primary data: ./data/stokes_primary_regr.csv
    ## Second arg: path to Broad repurposing hub data: ./data/stokes_broad_hub_smiles_unc.csv
    ##      this file contains a column containing the uncertainties.
    parser.add_argument('--smiles_paths', nargs='+', type=str, required=True,
                        help='Path to .csv files containing smiles strings (with header)')
    ## ./stokes_results
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to a directory where the t-SNE xy coordinates and plot will be saved')
    args = parser.parse_args()

    tsne_xy_coordinates(
        smiles_paths=args.smiles_paths,
        save_dir=args.save_dir
    )
