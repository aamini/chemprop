from argparse import ArgumentParser
import csv
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.cluster import KMeans
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.utils import makedirs
from chemprop.features import get_features_generator
from standardize_smiles import standardize_smiles


def cluster_zinc_molecules(data_path: str,
                           save_dir: str,
                           train_smiles_path: str,
                           train_hits_smiles_path: str,
                           num_clusters: int):
    # Load data
    with open(data_path) as f:
        data = list(csv.DictReader(f))

    with open(train_smiles_path) as f:
        train_smiles = {row['smiles'] for row in csv.DictReader(f)}

    with open(train_hits_smiles_path) as f:
        train_hits_smiles = {row['smiles'] for row in csv.DictReader(f)}

    # Standardize smiles
    train_smiles_standardized = {standardize_smiles(smiles) for smiles in train_smiles}
    train_hits_smiles_standardized = {standardize_smiles(smiles) for smiles in train_hits_smiles}

    for row in data:
        standardized = standardize_smiles(row['smiles'])
        row['in_train'] = standardized in train_smiles_standardized
        row['in_train_hit'] = standardized in train_hits_smiles_standardized

    # Compute Morgan fingerprints
    print('Computing Morgan fingerprints')
    morgan_generator = get_features_generator('morgan')
    morgans = [morgan_generator(datapoint['smiles']) for datapoint in tqdm(data, total=len(data))]

    # Perform clustering
    print('Clustering')
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_jobs=-1).fit(morgans)

    # Determine mapping from cluster index to datapoints in that cluster
    clusters = [[] for _ in range(num_clusters)]
    for cluster, datapoint in zip(kmeans.labels_, data):
        clusters[cluster].append(datapoint)

    # Save clustering
    print('Saving clusters and images')
    max_cluster_size = max(len(cluster) for cluster in clusters)
    cluster_table = []
    for cluster, cluster_data in tqdm(enumerate(clusters), total=len(clusters)):
        # Make cluster directory
        cluster_dir = os.path.join(save_dir, f'cluster_{cluster}')
        makedirs(cluster_dir)

        # Get smiles and zinc indices for molecules in cluster
        smiles, zinc_indices, in_train, in_train_hit, pred = zip(*[
            (
                datapoint['smiles'],
                datapoint['zinc_index'],
                datapoint['in_train'],
                datapoint['in_train_hit'],
                datapoint['50uMInhibition_Avg']
            )
            for datapoint in cluster_data
        ])

        # Add smiles and zinc indices to cluster table
        blanks = [''] * (max_cluster_size - len(smiles))
        cluster_table.append([f'cluster_{cluster}_smiles'] + list(smiles) + blanks)
        cluster_table.append([f'cluster_{cluster}_zinc_indices'] + list(zinc_indices) + blanks)
        cluster_table.append([f'cluster_{cluster}_in_train'] + list(in_train) + blanks)
        cluster_table.append([f'cluster_{cluster}_in_train_hit'] + list(in_train_hit) + blanks)
        cluster_table.append([f'cluster_{cluster}_50uMInhibition_Avg'] + list(pred) + blanks)

        # Add smiles and zinc indices to cluster table and save molecule images
        for smile, zinc_index in zip(smiles, zinc_indices):
            mol = Chem.MolFromSmiles(smile)
            Draw.MolToFile(mol, os.path.join(cluster_dir, f'{zinc_index}.png'))

    # Save cluster table
    cluster_table = np.array(cluster_table).T
    with open(os.path.join(save_dir, 'clusters.csv'), 'w') as f:
        writer = csv.writer(f)

        for row in cluster_table:
            writer.writerow(row)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file containing SMILES strings and zinc indices')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to directory where the clustering and PNG images of molecules will be saved')
    parser.add_argument('--train_smiles_path', type=str, required=True,
                        help='Path to CSV file containing the training smiles')
    parser.add_argument('--train_hits_smiles_path', type=str, required=True,
                        help='Path to CSV file containing the training hit smiles')
    parser.add_argument('--num_clusters', type=int, default=250,
                        help='Number of clusters')
    args = parser.parse_args()

    cluster_zinc_molecules(
        data_path=args.data_path,
        save_dir=args.save_dir,
        train_smiles_path=args.train_smiles_path,
        train_hits_smiles_path=args.train_hits_smiles_path,
        num_clusters=args.num_clusters
    )
