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


def generate_conformers(smiles: str,
                        num_conformers: int,
                        max_attempts: int,
                        prune_rms_threshold: float) -> Tuple[Chem.Mol, List[int]]:
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
                    num_conformers: int,
                    max_attempts: int,
                    prune_rms_threshold: float):
    os.makedirs(save_dir, exist_ok=True)

    pool = Pool()
    fnames = [fname for fname in os.listdir(data_dir) if fname.endswith('.txt')]
    generator = partial(
        generate_conformers,
        num_conformers=num_conformers,
        max_attempts=max_attempts,
        prune_rms_threshold=prune_rms_threshold
    )

    for fname in tqdm(fnames, total=len(fnames)):
        data_path = os.path.join(data_dir, fname)
        save_path = os.path.join(save_dir, f'{os.path.splitext(fname)[0]}.sdf')
        writer = Chem.SDWriter(save_path)

        smiles = get_smiles(data_path)

        for mol, ids in tqdm(pool.imap(generator, smiles), total=len(smiles)):
            for id in ids:
                writer.write(mol, confId=id)

        writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to a directory containing .txt files with smiles strings (with a header row)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to a directory where conformers will be saved as SDFs')
    parser.add_argument('--num_conformers', type=int, default=50,
                        help='Number of conformers to generate per molecule')
    parser.add_argument('--max_attempts', type=int, default=1000,
                         help='Maximum number of conformer attempts per molecule')
    parser.add_argument('--prune_rsm_threshold', type=float, default=0.1,
                         help='Pruning threshold for similar conformers')
    args = parser.parse_args()

    save_conformers(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        num_conformers=args.num_conformers,
        max_attempts=args.max_attempts,
        prune_rms_threshold=args.prune_rms_threshold
    )
