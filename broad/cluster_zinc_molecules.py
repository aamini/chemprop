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


def plot_kmeans_with_tsne(X: np.ndarray,
                          smiles: List[str],
                          cluster_labels: List[int],
                          cluster_center_indices: List[int],
                          top_indices: List[int],
                          num_clusters: int,
                          save_dir: str,
                          display_type: str):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(6.4 * 40, 4.8 * 40))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        color = plt.cm.rainbow(cluster_labels[i] / num_clusters)

        if display_type == 'points':
            plt.plot(X[i, 0], X[i, 1], marker='o', markersize=40, color=color)
        else:
            plt.text(X[i, 0], X[i, 1], str(cluster_labels[i]),
                     color=color, fontdict={'weight': 'bold', 'size': 80})

    # Display cluster number in cluster center if points display
    if display_type == 'points':
        # Find cluster centers
        clusters = [[] for _ in range(num_clusters)]
        for i in range(len(X)):
            clusters[cluster_labels[i]].append(X[i])
        cluster_centers = [np.mean(cluster, axis=0) for cluster in clusters]

        # Plot cluster centers
        for cluster, cluster_center in enumerate(cluster_centers):
            edgecolor = plt.cm.rainbow(cluster / num_clusters)
            facecolor = (*edgecolor[:-1], 0.2)  # make facecolor translucent

            x, y = cluster_center[0], cluster_center[1]

            plt.text(x, y, str(cluster),
                     ha="center", va="center",
                     bbox=dict(boxstyle="square",
                               edgecolor=edgecolor,
                               facecolor=facecolor),
                     fontdict={'weight': 'bold', 'size': 160})

    # Otherwise show images of molecules
    elif hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        if display_type == 'random':
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
        elif display_type in ['center', 'top']:
            indices = cluster_center_indices if display_type == 'center' else top_indices
            for i in indices:
                img = Draw.MolsToGridImage([Chem.MolFromSmiles(smiles[i])], molsPerRow=1, subImgSize=(400, 400))
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(img, cmap=plt.cm.gray_r),
                    X[i])
                ax.add_artist(imagebox)
        else:
            raise ValueError(f'Display type {display_type} not supported.')

    plt.xticks([]), plt.yticks([])
    plt.savefig(os.path.join(save_dir, f'kmeans_tsne_{display_type}.png'))


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
    for index, (cluster, datapoint) in enumerate(zip(kmeans.labels_, data)):
        datapoint['index'] = index
        clusters[cluster].append(datapoint)

    # Save clustering and images of cluster centers
    print('Saving clusters')
    cluster_center_indices, top_indices = [], []
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
        top_indices.append(top_datapoint['index'])

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
        cluster_center_indices.append(center_datapoint['index'])

        # Save image of cluster center
        mol = Chem.MolFromSmiles(center_datapoint['smiles'])
        Draw.MolToFile(mol, os.path.join(cluster_dir, f'cluster_{cluster}_center_{center_datapoint["zinc_index"]}.png'))

        # Pop unneeded keys
        for datapoint in cluster_data:
            del datapoint['index']
            del datapoint['morgan']

        # Save cluster data
        with open(os.path.join(cluster_dir, f'cluster_{cluster}.csv'), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cluster_data)

    print('Running T-SNE')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X = tsne.fit_transform(morgans)

    print('Plotting K-Means/T-SNE')
    smiles = [datapoint['smiles'] for datapoint in data]
    for display_type in ['points']:  # ['points', 'random', 'center', 'top']:
        plot_kmeans_with_tsne(
            X=X,
            smiles=smiles,
            cluster_labels=kmeans.labels_,
            cluster_center_indices=cluster_center_indices,
            top_indices=top_indices,
            num_clusters=num_clusters,
            save_dir=save_dir,
            display_type=display_type
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
