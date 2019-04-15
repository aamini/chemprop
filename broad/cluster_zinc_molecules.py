from argparse import ArgumentParser
import csv
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.cluster import KMeans
from tqdm import tqdm, trange

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.utils import makedirs
from chemprop.features import get_features_generator


def cluster_zinc_molecules(data_path: str,
                           save_dir: str,
                           num_clusters: int,
                           sort_column: str):
    # Load data
    with open(data_path) as f:
        data = list(csv.DictReader(f))
    fieldnames = list(data[0].keys())

    # Compute Morgan fingerprints
    print('Computing Morgan fingerprints')
    morgan_generator = get_features_generator('morgan')
    morgans = [morgan_generator(datapoint['smiles']) for datapoint in tqdm(data, total=len(data))]
    for datapoint, morgan in zip(data, morgans):
        datapoint['morgan'] = morgan

    # Perform clustering
    print('Clustering')
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_jobs=-1).fit(morgans)

    # Determine mapping from cluster index to datapoints in that cluster
    clusters = [[] for _ in range(num_clusters)]
    for cluster, datapoint in zip(kmeans.labels_, data):
        clusters[cluster].append(datapoint)

    # Save clustering and images of cluster centers
    print('Saving clusters')
    for cluster in trange(num_clusters):
        # Make cluster directory
        cluster_dir = os.path.join(save_dir, f'cluster_{cluster}')
        makedirs(cluster_dir)

        # Get cluster data and center
        cluster_data = clusters[cluster]
        cluster_center = kmeans.cluster_centers_[cluster]

        # Sort cluster data by prediction
        if sort_column is not None:
            cluster_data.sort(key=lambda datapoint: datapoint[sort_column], reverse=True)

        # Get top molecule
        top_datapoint = cluster_data[0]

        # Save image of top molecule
        mol = Chem.MolFromSmiles(top_datapoint['smiles'])
        Draw.MolToFile(mol, os.path.join(cluster_dir, f'center_{top_datapoint["zinc_index"]}.png'))

        # Find molecule neareest to cluster center
        center_datapoint, center_distance = None, float('inf')
        for datapoint in cluster_data:
            distance = np.linalg.norm(cluster_center - datapoint['morgan'])
            if distance < center_distance:
                center_distance = distance
                center_datapoint = datapoint

        # Save image of cluster center
        mol = Chem.MolFromSmiles(center_datapoint['smiles'])
        Draw.MolToFile(mol, os.path.join(cluster_dir, f'center_{center_datapoint["zinc_index"]}.png'))

        # Pop morgan vector
        for datapoint in cluster_data:
            del datapoint['morgan']

        # Save cluster data
        with open(os.path.join(cluster_dir, f'cluster_{cluster}.csv'), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cluster_data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file containing SMILES strings and zinc indices')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to directory where the clustering and PNG images of molecules will be saved')
    parser.add_argument('--num_clusters', type=int, default=50,
                        help='Number of clusters')
    parser.add_argument('--sort_column', type=str, default=None,
                        help='Column name by which to sort the data')
    args = parser.parse_args()

    cluster_zinc_molecules(
        data_path=args.data_path,
        save_dir=args.save_dir,
        num_clusters=args.num_clusters,
        sort_column=args.sort_column
    )
