from argparse import ArgumentParser
from copy import deepcopy
from collections import OrderedDict
import csv
import os
from typing import Callable, List

from rdkit import Chem
from rdkit.Chem import Draw


def save_top(all_clusters_dir: str, fname: str, top: List[List[OrderedDict]]):
    with open(os.path.join(all_clusters_dir, fname), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['cluster', 'ranking'] + list(top[0][0].keys()))
        writer.writeheader()

        for cluster, top_rows in enumerate(top):
            for ranking, row in enumerate(top_rows):
                row = deepcopy(row)

                row.update({'cluster': cluster})
                row.move_to_end('cluster', last=False)
                row.update(({'ranking': ranking}))
                row.move_to_end('ranking', last=False)

                writer.writerow(row)


def process_cluster(data: List[OrderedDict],
                    sort_fn: Callable,
                    cluster: int,
                    cluster_dir: str,
                    name: str,
                    num_mols_per_cluster: int) -> List[OrderedDict]:
    # Sort and extract top molecules
    data = sorted(data, key=sort_fn, reverse=True)
    top = data[:num_mols_per_cluster]

    # Draw top molecules
    for i, row in enumerate(top):
        mol = Chem.MolFromSmiles(row['smiles'])
        Draw.MolToFile(mol, os.path.join(cluster_dir, f'cluster_{cluster}_{name}_{i}_{row["zinc_index"]}.png'))

    return top


def top_in_cluster(all_clusters_dir: str, num_mols_per_cluster: int):
    top_pred, top_ratio_train, top_ratio_antibiotics = [], [], []

    cluster = 0
    cluster_dir = os.path.join(all_clusters_dir, f'cluster_{cluster}')

    while os.path.exists(cluster_dir):
        # Load cluster data
        with open(os.path.join(cluster_dir, f'cluster_{cluster}.csv')) as f:
            data = list(csv.DictReader(f))

        # Top by prediction
        top_pred.append(process_cluster(
            data=data,
            sort_fn=lambda row: float(row['50uMInhibition_Avg']),
            cluster=cluster,
            cluster_dir=cluster_dir,
            name='top_pred',
            num_mols_per_cluster=num_mols_per_cluster
        ))

        # Top by ratio of pred to tanimoto to train
        top_ratio_train.append(process_cluster(
            data=data,
            sort_fn=lambda row: float(row['50uMInhibition_Avg']) / float(row['train_set_tanimoto_sim']),
            cluster=cluster,
            cluster_dir=cluster_dir,
            name='top_ratio_to_train',
            num_mols_per_cluster=num_mols_per_cluster
        ))

        # Top by ratio of pred to tanimoto to antibiotics
        top_ratio_antibiotics.append(process_cluster(
            data=data,
            sort_fn=lambda row: float(row['50uMInhibition_Avg']) / float(row['antibiotics_tanimoto_sim']),
            cluster=cluster,
            cluster_dir=cluster_dir,
            name='top_ratio_to_antibiotics',
            num_mols_per_cluster=num_mols_per_cluster
        ))

        # Increment cluster count
        cluster += 1
        cluster_dir = os.path.join(all_clusters_dir, f'cluster_{cluster}')

    # Save top molecules
    save_top(
        all_clusters_dir=all_clusters_dir,
        fname='top_pred.csv',
        top=top_pred
    )

    save_top(
        all_clusters_dir=all_clusters_dir,
        fname='top_ratio_to_train.csv',
        top=top_ratio_train
    )

    save_top(
        all_clusters_dir=all_clusters_dir,
        fname='top_ratio_to_antibiotics.csv',
        top=top_ratio_antibiotics
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--all_clusters_dir', type=str, required=True,
                        help='Directory containing directories labelled cluster_0, cluster_1, ...')
    parser.add_argument('--num_mols_per_cluster', type=int, default=2,
                        help='How many of the top molecules to extract per cluster')
    args = parser.parse_args()

    top_in_cluster(
        all_clusters_dir=args.all_clusters_dir,
        num_mols_per_cluster=args.num_mols_per_cluster
    )
