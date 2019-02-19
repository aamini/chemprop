from argparse import ArgumentParser
from collections import defaultdict
import csv
from functools import partial
from multiprocessing import Pool
import os
import random
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import rdFMCS, rdMolAlign, rdShapeHelpers
from tqdm import tqdm


def load_conformers_from_file(sdf_path: str,
                              sample_prob: float,
                              print_frequency: int = 10000) -> List[Chem.Mol]:
    random.seed(0)

    conformers = []
    for i, mol in enumerate(Chem.SDMolSupplier(sdf_path)):
        if i % print_frequency == 0:
            print(f'{i:,}')

            # TODO: remove
            if i > 0:
                break
        if random.random() < sample_prob:
            conformers.append(mol)

    return conformers


def load_conformers_from_dir(sdf_dir: str,
                             sample_prob: float,
                             print_frequency: int) -> List[Chem.Mol]:
    sdf_paths = [os.path.join(sdf_dir, fname) for fname in os.listdir(sdf_dir) if fname.endswith('.sdf')]
    load_conformers_fn = partial(load_conformers_from_file, sample_prob=sample_prob, print_frequency=print_frequency)

    # Load conformers and construct smiles to conformers dictionaries
    with Pool() as pool:
        conformers = [conformer for conformers in tqdm(pool.imap(load_conformers_fn, sdf_paths), total=len(sdf_paths))
                      for conformer in conformers]

    return conformers


def compute_3d_distance(mol_pair: Tuple[Chem.Mol, Chem.Mol]) -> Tuple[Tuple[str, str], float]:
    mol1, mol2 = mol_pair
    smiles1, smiles2 = Chem.MolToSmiles(mol1), Chem.MolToSmiles(mol2)

    try:
        # TODO: should we be doing ringMatchesRingOnly?
        mcs = rdFMCS.FindMCS([mol1, mol2], ringMatchesRingOnly=True)
        core = Chem.MolFromSmarts(mcs.smartsString)
        match1, match2 = mol1.GetSubstructMatch(core), mol2.GetSubstructMatch(core)
        rdMolAlign.AlignMol(mol1, mol2, atomMap=list(zip(match1, match2)))
        distance = rdShapeHelpers.ShapeTanimotoDist(mol1, mol2)

        return (smiles1, smiles2), distance
    except Exception as e:
        print(e)
        return (smiles1, smiles2), 1.0


def construct_clusters(conformers: List[Chem.Mol],
                       num_clusters: int) -> Dict[Tuple[str, str], bool]:
    import numpy as np
    from itertools import combinations
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.misc import comb

    # Convert to smiles
    smiles = [Chem.MolToSmiles(mol) for mol in conformers]

    # Compute distances between pairs of conformers
    mol_pairs = combinations(conformers, 2)
    num_mol_pairs = int(comb(len(conformers), 2))
    with Pool() as pool:
        smiles_pair_distances = list(tqdm(pool.imap(compute_3d_distance, mol_pairs), total=num_mol_pairs))
    smiles_pairs, distances = zip(*smiles_pair_distances)

    # Perform clustering
    link = linkage(np.array(distances))
    clusters = fcluster(link, t=num_clusters, criterion='maxclust')

    # Determine which pairs of smiles have at least one conformer in the same cluster
    smiles_clusters = defaultdict(bool)
    for i, j in combinations(range(len(smiles)), 2):
        same_cluster = clusters[i] == clusters[j]
        smiles_pair = (smiles[i], smiles[j])
        smiles_clusters[smiles_pair] = max(smiles_clusters[smiles_pair], same_cluster)

    return smiles_clusters


def build_conformer_pairs(sdf_dir: str,
                          save_path: str,
                          num_clusters: int,
                          sample_prob: float,
                          print_frequency: int):
    conformers = load_conformers_from_dir(sdf_dir, sample_prob, print_frequency)
    smiles_clusters = construct_clusters(conformers, num_clusters)

    num_pairs = len(smiles_clusters)
    num_overlapping_pairs = sum(smiles_clusters.values())

    print(f'Number of pairs = {num_pairs}')
    print(f'Number of overlapping pairs = {num_overlapping_pairs}')
    print(f'Percent of overlapping pairs = {num_overlapping_pairs / num_pairs * 100}%')

    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles_1', 'smiles_2', 'overlap'])
        for (smiles1, smiles2), overlap in smiles_clusters.items():
            writer.writerow([smiles1, smiles2, overlap])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sdf_dir', type=str, required=True,
                        help='Path to directory containing .sdf files with conformers')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to .csv file where pairs will be saved')
    parser.add_argument('--num_clusters', type=int, default=1000,
                        help='Number of clusters')
    parser.add_argument('--sample_prob', type=float, default=1,
                        help='Probability with which each conformer is sampled')
    parser.add_argument('--print_frequency', type=int, default=10000,
                        help='Print frequency when loading conformers from disk')
    args = parser.parse_args()

    build_conformer_pairs(
        sdf_dir=args.sdf_dir,
        save_path=args.save_path,
        num_clusters=args.num_clusters,
        sample_prob=args.sample_prob,
        print_frequency=args.print_frequency
    )
