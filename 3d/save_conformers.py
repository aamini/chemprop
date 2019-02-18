"""Script for generating and saving conformers for molecules."""

from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from chemprop.data.utils import get_smiles


# Loading from SDF:
# supp = Chem.SDMolSupplier('AAAA.sdf')
# for mol in supp:
#     conformer = mol.GetConformer()
#     for i in range(mol.GetNumAtoms():
#         pos = conformer.GetAtomPosition(i)
#     smiles = Chem.MolToSmiles(mol)


# Computing distance:
# rms = Chem.rdMolAlign.AlignMol(mol1, mol2)
# tani = Chem.rdShapeHelpers.ShapeTanimotoDist(mol1, mol2)


def generate_conformers(smiles: str,
                        num_conformers: int,
                        max_attempts: int,
                        prune_rms_threshold: float) -> Tuple[Chem.Mol, List[int]]:
    """
    Generates multiple conformers for a molecule.

    :param smiles: A SMILES string for a molecule.
    :param num_conformers: Number of conformers to find for each molecule.
    :param max_attempts: Maximum number of attempts to find a conformer.
    :param prune_rms_threshold: Pruning threshold for similar conformers.
    :return: A tuple containing the RDKit molecule (now containing conformers) and a list of conformer ids.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    conformers = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=num_conformers,
        maxAttempts=max_attempts,
        pruneRmsThresh=prune_rms_threshold
    )
    ids = list(conformers)

    return mol, ids


def save_conformers(data_dir: str,
                    save_dir: str,
                    save_frequency: int,
                    num_conformers: int,
                    max_attempts: int,
                    prune_rms_threshold: float,
                    sequential: bool):
    """
    Computes and saves multiple conformers for each molecule as an SDF file.

    :param data_dir: Directory containing .txt files with SMILES strings.
    :param save_dir: Directory where .sdf files containing molecules/conformers will be saved.
    :param save_frequency: Number of molecules saved to each SDF file.
    :param num_conformers: Number of conformers to find for each molecule.
    :param max_attempts: Maximum number of attempts to find a conformer.
    :param prune_rms_threshold: Pruning threshold for similar conformers.
    :param sequential: Whether to compute conformers sequentially rather than in parallel.
    """
    os.makedirs(save_dir, exist_ok=True)

    fnames = sorted([fname for fname in os.listdir(data_dir) if fname.endswith('.txt')])
    generator = partial(
        generate_conformers,
        num_conformers=num_conformers,
        max_attempts=max_attempts,
        prune_rms_threshold=prune_rms_threshold
    )
    map_fn = map if sequential else Pool().imap
    sdf_num = mol_count = 0
    writer = None

    for fname in tqdm(fnames, total=len(fnames)):
        data_path = os.path.join(data_dir, fname)
        smiles = get_smiles(data_path)

        for mol, ids in tqdm(map_fn(generator, smiles), total=len(smiles)):
            if mol_count % save_frequency == 0:
                save_path = os.path.join(save_dir, f'{sdf_num * save_frequency}-{(sdf_num + 1) * save_frequency - 1}.sdf')
                if writer is not None:
                    writer.close()
                writer = Chem.SDWriter(save_path)
                sdf_num += 1

            for id in ids:
                writer.write(mol, confId=id)

            mol_count += 1

        writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to a directory containing .txt files with smiles strings (with a header row)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to a directory where conformers will be saved as SDFs')
    parser.add_argument('--save_frequency', type=int, default=10000,
                        help='Number of molecules saved to each SDF file')
    parser.add_argument('--num_conformers', type=int, default=50,
                        help='Number of conformers to generate per molecule')
    parser.add_argument('--max_attempts', type=int, default=1000,
                        help='Maximum number of conformer attempts per molecule')
    parser.add_argument('--prune_rms_threshold', type=float, default=0.1,
                        help='Pruning threshold for similar conformers')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Whether to compute conformers sequentially rather than in parallel')
    args = parser.parse_args()

    save_conformers(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        save_frequency=args.save_frequency,
        num_conformers=args.num_conformers,
        max_attempts=args.max_attempts,
        prune_rms_threshold=args.prune_rms_threshold,
        sequential=args.sequential
    )
