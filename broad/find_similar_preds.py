from argparse import ArgumentParser
import csv
import sys
sys.path.append('../')

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from chemprop.data.utils import get_data
from chemprop.features import morgan_fingerprint
from chemprop.nn_utils import compute_molecule_vectors
from chemprop.train.predict import predict
from chemprop.utils import load_checkpoint


def find_similar_preds(test_path: str,
                       train_path: str,
                       checkpoint_path: str,
                       distance_measure: str,
                       save_path: str,
                       num_neighbors: int,
                       batch_size: int):
    print('Loading data')
    test_data, train_data = get_data(test_path), get_data(train_path)
    test_smiles, train_smiles = test_data.smiles(), train_data.smiles()
    train_smiles_set = set(train_smiles)

    print('Computing morgan fingerprints')
    test_morgans = np.array([morgan_fingerprint(smiles) for smiles in tqdm(test_data.smiles(), total=len(test_data))])
    train_morgans = np.array([morgan_fingerprint(smiles) for smiles in tqdm(train_data.smiles(), total=len(train_data))])

    print('Loading model')
    model = load_checkpoint(checkpoint_path)

    print('Predicting')
    test_preds = predict(model=model, data=test_data, batch_size=batch_size)

    print('Computing molecule vectors')
    test_vecs = compute_molecule_vectors(model=model, data=test_data, batch_size=batch_size)
    train_vecs = compute_molecule_vectors(model=model, data=train_data, batch_size=batch_size)
    test_vecs, train_vecs = np.array(test_vecs), np.array(train_vecs)

    print('Computing distances')
    test_train_morgan_dist = cdist(test_morgans, train_morgans, metric='jaccard')
    test_train_vec_dist = cdist(test_vecs, train_vecs, metric='cosine')

    if distance_measure == 'vec':
        dist_matrix = test_train_vec_dist
    elif distance_measure == 'morgan':
        dist_matrix = test_train_morgan_dist
    else:
        raise ValueError(f'Distance measure "{distance_measure}" not supported.')

    print('Finding neighbors')
    neighbors = []
    for test_index in range(len(test_data)):
        # Find the num_neighbors molecules in the training set which are most similar to the test molecule
        nearest_train = np.argsort(dist_matrix[test_index])[:num_neighbors]

        # Get the distances from test molecule to the nearest train molecules
        nearest_train_vec_dists = test_train_vec_dist[test_index][nearest_train]
        nearest_train_morgan_dists = test_train_morgan_dist[test_index][nearest_train]

        # Build dictionary with distance info
        neighbor = {
            'test_smiles': test_smiles[test_index],
            'in_train': test_smiles[test_index] in train_smiles_set,
            'test_pred': test_preds[test_index][0],
            f'train_{num_neighbors}_avg_vec_cosine_dist': np.mean(nearest_train_vec_dists),
            f'train_{num_neighbors}_avg_morgan_jaccard_dist': np.mean(nearest_train_morgan_dists)
        }
        for i, train_index in enumerate(nearest_train):
            neighbor[f'train_smiles_{i + 1}'] = train_smiles[train_index]
            neighbor[f'train_vec_cosine_dist_{i + 1}'] = nearest_train_vec_dists[i]
            neighbor[f'train_morgan_jaccard_dist_{i + 1}'] = nearest_train_morgan_dists[i]

        neighbors.append(neighbor)

    # Sort by test prediction
    neighbors.sort(key=lambda neighbor: neighbor['test_pred'], reverse=True)

    # Construct fieldnames for save file
    fieldnames = [
        'test_smiles',
        'in_train',
        'test_pred',
        f'train_{num_neighbors}_avg_vec_cosine_dist',
        f'train_{num_neighbors}_avg_morgan_jaccard_dist'
    ]
    for i in range(num_neighbors):
        fieldnames += [
            f'train_smiles_{i + 1}',
            f'train_vec_cosine_dist_{i + 1}',
            f'train_morgan_jaccard_dist_{i + 1}'
        ]

    print('Saving distances')
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
    parser.add_argument('--distance_measure', type=str, choices=['vec', 'morgan'], default='vec',
                        help='Distance measure to use to find nearest neighbors in train set')
    parser.add_argument('--num_neighbors', type=int, default=5,
                        help='Number of neighbors to search for each molecule')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size when making predictions')
    args = parser.parse_args()

    find_similar_preds(
        test_path=args.test_path,
        train_path=args.train_path,
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        distance_measure=args.distance_measure,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size
    )
