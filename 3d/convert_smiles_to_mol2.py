"""Script for converting SMILES files to mol2 files."""

from argparse import ArgumentParser
from multiprocessing import Pool
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import List, Tuple

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_smiles


def convert_smiles_chunk_to_mol2(smiles_and_path: Tuple[List[str], str]):
    """
    Converts a chunk of SMILES strings and saves to a mol2 file.

    :param smiles_and_path: A tuple containing a list of smiles and a path to save the mol2 file.
    """
    smiles, save_path = smiles_and_path

    with NamedTemporaryFile(suffix='.smi') as f_smi:
        f_smi.writelines(smiles)
        subprocess.run(['babel', '-ismi', f_smi.name, '-omol2', save_path])


def convert_smiles_to_mol2(data_dir: str,
                           save_dir: str,
                           save_frequency: int,
                           sequential: bool):
    """
    Converts files of SMILES strings to mol2 files.

    :param data_dir: Directory containing .txt files with SMILES strings.
    :param save_dir: Directory where .mol2 files containing molecules/conformers will be saved.
    :param save_frequency: Number of molecules saved to each mol2 file.
    :param sequential: Whether to convert sequentially rather than in parallel.
    """
    os.makedirs(save_dir, exist_ok=True)

    data_paths = sorted([os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.txt')])
    map_fn = map if sequential else Pool().imap

    smiles = [smile for smiles in tqdm(map_fn(get_smiles, data_paths), total=len(data_paths)) for smile in smiles]
    smiles_chunks = (smiles[i:i + save_frequency] for i in range(0, len(smiles), save_frequency))
    save_paths = (os.path.join(save_dir, f'{i}-{min(i + save_frequency, len(smiles))}.mol2')
                  for i in range(0, len(smiles), save_frequency))
    smiles_and_paths = zip(smiles_chunks, save_paths)

    for _ in tqdm(map_fn(convert_smiles_chunk_to_mol2, smiles_and_paths), total=len(smiles) // save_frequency + 1):
        pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to a directory containing .txt files with smiles strings (with a header row)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to a directory where mol2 files will be saved')
    parser.add_argument('--save_frequency', type=int, default=10000,
                        help='Number of molecules saved to each mol2 file')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Whether to compute conformers sequentially rather than in parallel')
    args = parser.parse_args()

    convert_smiles_to_mol2(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        save_frequency=args.save_frequency,
        sequential=args.sequential
    )
