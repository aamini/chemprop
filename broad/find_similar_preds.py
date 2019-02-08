from argparse import ArgumentParser
import csv
import sys
sys.path.append('../')

import numpy as np
from scipy.spatial.distance import cdist

from chemprop.data.utils import get_data
from chemprop.features import morgan_fingerprint
from chemprop.nn_utils import compute_molecule_vectors
from chemprop.train.predict import predict
from chemprop.utils import load_checkpoint


def find_similar_preds(test_path: str,
                       train_path: str,
                       checkpoint_path: str,
                       save_path: str,
                       num_neighbors: int,
                       batch_size: int):
    print('Loading data')
    test_data, train_data = get_data(test_path), get_data(train_path)
    test_smiles, train_smiles = test_data.smiles(), train_data.smiles()

    print('Computing morgan fingerprints')
    test_morgans = np.array([morgan_fingerprint(smiles) for smiles in test_data.smiles()])
    train_morgans = np.array([morgan_fingerprint(smiles) for smiles in train_data.smiles()])

    print('Loading model')
    model = load_checkpoint(checkpoint_path)

    print('Predicting')
    test_preds = predict(model=model, data=test_data, batch_size=batch_size)
    # train_preds = predict(model=model, data=train_data, batch_size=batch_size)

    print('Computing molecule vectors')
    test_vecs = compute_molecule_vectors(model=model, data=test_data, batch_size=batch_size)
    train_vecs = compute_molecule_vectors(model=model, data=train_data, batch_size=batch_size)
    test_vecs, train_vecs = np.array(test_vecs), np.array(train_vecs)

    print('Computing distances')
    test_train_morgan_dist = cdist(test_morgans, train_morgans, metric='jaccard')
    test_train_vec_dist = cdist(test_vecs, train_vecs, metric='cosine')

    print('Finding neighbors')
    neighbors = []
    for test_index in range(len(test_data)):
        # Find the num_neighbors molecules in the training set which are
        # most similar to the test molecule by molecule vector distance
        nearest_train_by_vec_dist = np.argsort(test_train_vec_dist[test_index])[:num_neighbors]

        # Get the distances from test molecule to the nearest train molecules
        nearest_train_vec_dists = test_train_vec_dist[test_index][nearest_train_by_vec_dist]
        nearest_train_morgan_dists = test_train_morgan_dist[test_index][nearest_train_by_vec_dist]

        # Build dictionary with distance info
        neighbor = {
            'test_smiles': test_smiles[test_index],
            'test_pred': test_preds[test_index][0],
            f'train_{num_neighbors}_avg_vec_dist': np.mean(nearest_train_vec_dists),
            f'train_{num_neighbors}_avg_morgan_dist': np.mean(nearest_train_morgan_dists)
        }
        for i, train_index in enumerate(nearest_train_by_vec_dist):
            neighbor[f'train_smiles_{i + 1}'] = train_smiles[train_index]
            neighbor[f'train_vec_dist_{i + 1}'] = nearest_train_vec_dists[i]
            neighbor[f'train_morgan_dist_{i + 1}'] = nearest_train_morgan_dists[i]

        neighbors.append(neighbor)
    neighbors.sort(key=lambda neighbor: neighbor['test_pred'], reverse=True)

    print('Saving distances')
    fieldnames = [
        'test_smiles',
        'test_pred',
        f'train_{num_neighbors}_avg_vec_dist',
        f'train_{num_neighbors}_avg_morgan_dist'
    ]
    for i in range(num_neighbors):
        fieldnames += [
            f'train_smiles_{i + 1}',
            f'train_vec_dist_{i + 1}',
            f'train_morgan_dist_{i + 1}'
        ]
    with open(save_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for neighbor in neighbors:
            writer.writerow(neighbor)


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
    parser.add_argument('--num_neighbors', type=int, default=5,
                        help='Number of neighbors to search for each molecule')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size when making predictions')
    args = parser.parse_args()

    find_similar_preds(
        args.test_path,
        args.train_path,
        args.checkpoint_path,
        args.save_path,
        args.num_neighbors,
        args.batch_size
    )
