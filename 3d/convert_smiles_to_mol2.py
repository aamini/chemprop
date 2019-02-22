"""Script for converting SMILES files to mol2 files."""

from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import List, Generator

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_smiles
from chemprop.utils import makedirs


def convert_smile_to_mol2(smiles: str, save_dir: str):
    """
    Converts a chunk of SMILES strings and saves to a mol2 file.

    :param smiles: A smiles string.
    :param save_dir: Directory where mol2 will be saved.
    """
    save_path = os.path.join(save_dir, f'{smiles}.mol2')

    with NamedTemporaryFile(mode='w') as f_smi:
        f_smi.write(smiles)
        f_smi.flush()
        with open(os.devnull, "w") as null:
            subprocess.run(['babel', '-ismi', f_smi.name, '-omol2', save_path], stdout=null, stderr=subprocess.STDOUT)


def smiles_generator(data_paths: List[str], max_num_molecules: int) -> Generator[str, None, None]:
    """
    Yields smiles strings from multiple data files.

    :param data_paths: A list of paths to CSV files containing smiles strings (with headers).
    :param max_num_molecules: The maximum number of molecules to yield.
    :return: A generator which yields smiles strings.
    """
    max_num_molecules = max_num_molecules or float('inf')
    num_molecules = 0
    for data_path in data_paths:
        for smiles in get_smiles(data_path):
            if num_molecules >= max_num_molecules:
                return

            yield smiles
            num_molecules += 1


def convert_smiles_to_mol2(data_dir: str,
                           save_dir: str,
                           max_num_molecules: int,
                           sequential: bool):
    """
    Converts files of SMILES strings to mol2 files.

    :param data_dir: Directory containing .txt files with SMILES strings.
    :param save_dir: Directory where .mol2 files containing molecules/conformers will be saved.
    :param max_num_molecules: Maximum number of molecules to convert.
    :param sequential: Whether to convert sequentially rather than in parallel.
    """
    makedirs(save_dir)

    data_paths = sorted([os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.txt')])
    smiles = smiles_generator(data_paths, max_num_molecules)

    map_fn = map if sequential else Pool().imap
    convert_fn = partial(convert_smile_to_mol2, save_dir=save_dir)

    for _ in tqdm(map_fn(convert_fn, smiles), total=max_num_molecules):
        pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to a directory containing .txt files with smiles strings (with a header row)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to a directory where mol2 files will be saved')
    parser.add_argument('--max_num_molecules', type=int, default=None,
                        help='Maximum number of molecules to convert')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Whether to compute conformers sequentially rather than in parallel')
    args = parser.parse_args()

    convert_smiles_to_mol2(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        max_num_molecules=args.max_num_molecules,
        sequential=args.sequential
    )
