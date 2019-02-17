from argparse import ArgumentParser
import csv
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_data
from chemprop.features import morgan_fingerprint
from chemprop.nn_utils import compute_molecule_vectors
from chemprop.utils import load_checkpoint


def distance_matrix(test_path: str,
                    train_path: str,
                    checkpoint_path: str,
                    distance_measure: str,
                    save_path: str,
                    batch_size: int):
    print('Loading data')
    test_data, train_data = get_data(test_path), get_data(train_path)
    test_smiles, train_smiles = test_data.smiles(), train_data.smiles()

    print('Computing morgan fingerprints')
    test_morgans = np.array([morgan_fingerprint(smiles) for smiles in tqdm(test_data.smiles(), total=len(test_data))])
    train_morgans = np.array([morgan_fingerprint(smiles) for smiles in tqdm(train_data.smiles(), total=len(train_data))])

    print('Loading model')
    model = load_checkpoint(checkpoint_path)

    print('Computing molecule vectors')
    test_vecs = compute_molecule_vectors(model=model, data=test_data, batch_size=batch_size)
    train_vecs = compute_molecule_vectors(model=model, data=train_data, batch_size=batch_size)
    test_vecs, train_vecs = np.array(test_vecs), np.array(train_vecs)

    print('Computing distances')
    if distance_measure == 'vec':
        dist_matrix = cdist(test_vecs, train_vecs, metric='cosine')
    elif distance_measure == 'morgan':
        dist_matrix = cdist(test_morgans, train_morgans, metric='jaccard')
    else:
        raise ValueError(f'Distance measure "{distance_measure}" not supported.')

    print('Saving distances')
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([f'test\\train ({distance_measure} distance)'] + train_smiles)
        for smiles, dists in zip(test_smiles, dist_matrix):
            writer.writerow([smiles] + dists.tolist())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to CSV file with test set of molecules')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to CSV file with train set of molecules')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to .pt file containing a model checkpoint')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to CSV file where similar molecules will be saved')
    parser.add_argument('--distance_measure', type=str, choices=['vec', 'morgan'], default='vec',
                        help='Distance measure to use to find nearest neighbors in train set')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size when making predictions')
    args = parser.parse_args()

    distance_matrix(
        test_path=args.test_path,
        train_path=args.train_path,
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        distance_measure=args.distance_measure,
        batch_size=args.batch_size
    )