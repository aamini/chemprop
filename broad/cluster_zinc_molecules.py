from argparse import ArgumentParser
import csv
import os
import sys
from typing import List

from matplotlib import offsetbox
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tqdm import tqdm, trange

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.utils import makedirs
from chemprop.features import get_features_generator


def plot_kmeans_with_tsne(smiles: List[str],
                          morgans: List[np.ndarray],
                          cluster_labels: List[int],
                          num_clusters: int,
                          save_dir: str):
    print('Running T-SNE')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X = tsne.fit_transform(morgans)

    print('Plotting K-Means/T-SNE')
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(6.4 * 40, 4.8 * 40))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(cluster_labels[i]),
                 color=plt.cm.rainbow(cluster_labels[i] / num_clusters),
                 fontdict={'weight': 'bold', 'size': 80})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            img = Draw.MolsToGridImage([Chem.MolFromSmiles(smiles[i])], molsPerRow=1, subImgSize=(400, 400))
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(img, cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    plt.savefig(os.path.join(save_dir, 'kmeans_tsne.png'))


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
        Draw.MolToFile(mol, os.path.join(cluster_dir, f'cluster_{cluster}_top_{top_datapoint["zinc_index"]}.png'))

        # Find molecule neareest to cluster center
        center_datapoint, center_distance = None, float('inf')
        for datapoint in cluster_data:
            distance = np.linalg.norm(cluster_center - datapoint['morgan'])
            if distance < center_distance:
                center_distance = distance
                center_datapoint = datapoint

        # Save image of cluster center
        mol = Chem.MolFromSmiles(center_datapoint['smiles'])
        Draw.MolToFile(mol, os.path.join(cluster_dir, f'cluster_{cluster}_center_{center_datapoint["zinc_index"]}.png'))

        # Pop morgan vector
        for datapoint in cluster_data:
            del datapoint['morgan']

        # Save cluster data
        with open(os.path.join(cluster_dir, f'cluster_{cluster}.csv'), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cluster_data)

    # Plot kmeans
    plot_kmeans_with_tsne(
        smiles=[datapoint['smiles'] for datapoint in data],
        morgans=morgans,
        cluster_labels=kmeans.labels_,
        num_clusters=num_clusters,
        save_dir=save_dir
    )


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
