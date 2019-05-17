from argparse import ArgumentParser
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_smiles
from chemprop.features import get_features_generator
from chemprop.utils import makedirs


su3327_smiles = 'Nc1nnc(Sc2ncc(s2)[N+]([O-])=O)s1'


def compare_datasets_tsne(smiles_path_1: str, smiles_path_2: str, save_path: str):
    # Load the two smiles datasets
    print('Loading data')
    smiles_1, smiles_2 = get_smiles(smiles_path_1), get_smiles(smiles_path_2)
    smiles = smiles_1 + smiles_2
    colors = ['blue'] * len(smiles_1) + ['red'] * len(smiles_2)

    su3327_index = smiles.index(su3327_smiles)

    # Compute Morgan fingerprints
    print('Computing Morgan fingerprints')
    morgan_generator = get_features_generator('morgan')
    morgans = [morgan_generator(smile) for smile in tqdm(smiles, total=len(smiles))]

    print('Running t-SNE')
    tsne = TSNE(n_components=2, init='pca', random_state=0, metric='jaccard')
    X = tsne.fit_transform(morgans)

    print('Plotting t-SNE')
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(6.4 * 10, 4.8 * 10))

    for i, (x, color) in enumerate(zip(X, colors)):
        if i == su3327_index:
            plt.plot(x[0], x[1], marker='*', markersize=80, color='green')
        else:
            plt.plot(x[0], x[1], marker='o', markersize=10, color=color)

    plt.xticks([]), plt.yticks([])

    makedirs(save_path, isfile=True)
    plt.savefig(os.path.join(save_path))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--smiles_path_1', type=str, required=True,
                        help='Path to a .csv file containing smiles strings (with header)')
    parser.add_argument('--smiles_path_2', type=str, required=True,
                        help='Path to a .csv file containing smiles strings (with header)')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to a .png file where the t-SNE plot will be saved')
    args = parser.parse_args()

    compare_datasets_tsne(
        smiles_path_1=args.smiles_path_1,
        smiles_path_2=args.smiles_path_2,
        save_path=args.save_path
    )
