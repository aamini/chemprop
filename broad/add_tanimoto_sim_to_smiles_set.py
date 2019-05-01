from argparse import ArgumentParser
import csv

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from chemprop.data.utils import get_smiles
from chemprop.features import get_features_generator


def add_tanimoto_sim_to_smiles_set(data_path: str,
                                   smiles_path: str,
                                   header: bool,
                                   save_path: str,
                                   new_column_name: str):
    print('Load data')
    with open(data_path) as f:
        data = list(csv.DictReader(f))

    print('Load smiles')
    smiles = list(set(get_smiles(smiles_path, header=header)))

    print('Computing morgan fingerprints')
    morgan_fingerprint = get_features_generator('morgan')
    data_morgans = np.array([morgan_fingerprint(row['smiles']) for row in tqdm(data, total=len(data))])
    smiles_morgans = np.array([morgan_fingerprint(smile) for smile in tqdm(smiles, total=len(smiles))])
    sims = 1 - cdist(data_morgans, smiles_morgans, metric='jaccard')
    max_sim_indices = np.argmax(sims, axis=1)

    print('Finding min distance smiles')
    for i, row in enumerate(data):
        max_sim_index = max_sim_indices[i]
        row[f'{new_column_name}_tanimoto_sim'] = sims[i, max_sim_index]
        row[f'{new_column_name}_neighbor'] = smiles[max_sim_index]

    print('Saving')
    with open(save_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()

        for row in data:
            writer.writerow(row)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to a data .csv with one column called "smiles"')
    parser.add_argument('--smiles_path', type=str, required=True,
                        help='Path to a .txt file containing smiles strings to compared to')
    parser.add_argument('--header', action='store_true', default=False,
                        help='Whether smiles_path file has a header')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to a .csv file where the data with tanimoto similarities will be saved')
    parser.add_argument('--new_column_name', type=str, default='train_set',
                        help='Name of new column containing tanimoto similarity which will be added to the csv')
    args = parser.parse_args()

    add_tanimoto_sim_to_smiles_set(
        data_path=args.data_path,
        smiles_path=args.smiles_path,
        header=args.header,
        save_path=args.save_path,
        new_column_name=args.new_column_name
    )
