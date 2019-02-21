"""Script for converting SMILES files to mol2 files."""

from argparse import ArgumentParser
from multiprocessing import Pool
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import List, Generator, Tuple

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_smiles


def convert_smiles_chunk_to_mol2(smiles_and_path: Tuple[List[str], str]):
    """
    Converts a chunk of SMILES strings and saves to a mol2 file.

    :param smiles_and_path: A tuple containing a list of smiles and a path to save the mol2 file.
    """
    smiles, save_path = smiles_and_path

    with NamedTemporaryFile(suffix='.smi', mode='w') as f_smi:
        for smile in smiles:
            f_smi.write(smile + '\n')
        f_smi.flush()
        subprocess.run(['babel', '-ismi', f_smi.name, '-omol2', save_path])


def smiles_and_path_chunker(smiles: Generator[str, None, None],
                            chunk_size: int,
                            save_dir: str) -> Generator[Tuple[List[str], str], None, None]:
    """
    Chunks smiles and returns chunks and mol2 save paths.

    :param smiles: A generator of smiles strings.
    :param chunk_size: The number of smiles in each chunk.
    :param save_dir: The directory where the save paths should save to.
    :return: A generator which generators tuples of a list of smiles and a save path.
    """
    start_index = end_index = 0
    chunk = []
    for smile in smiles:
        if start_index >= 100000:
            break

        chunk.append(smile)

        if len(chunk) == chunk_size:
            save_path = os.path.join(save_dir, f'{start_index}-{end_index}.mol2')
            yield chunk, save_path

            end_index += 1
            start_index = end_index
            chunk = []
        else:
            end_index += 1

    save_path = os.path.join(save_dir, f'{start_index}-{end_index}.mol2')
    yield chunk, save_path


def get_num_lines(path: str) -> int:
    """
    Get the number of lines in a file.

    :param path: Path to file.
    :return: The number of lines in the file.
    """
    with open(path) as f:
        length = sum(1 for _ in f)
    return length


def convert_smiles_to_mol2(data_dir: str,
                           save_dir: str,
                           chunk_size: int,
                           max_num_molecules: int,
                           sequential: bool):
    """
    Converts files of SMILES strings to mol2 files.

    :param data_dir: Directory containing .txt files with SMILES strings.
    :param save_dir: Directory where .mol2 files containing molecules/conformers will be saved.
    :param chunk_size: Number of molecules saved to each mol2 file.
    :param max_num_molecules: 
    :param sequential: Whether to convert sequentially rather than in parallel.
    """
    os.makedirs(save_dir, exist_ok=True)

    map_fn = map if sequential else Pool().imap
    data_paths = sorted([os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.txt')])

    smiles = (smile for data_path in tqdm(data_paths, total=len(data_paths)) for smile in get_smiles(data_path))
    smiles_and_paths = smiles_and_path_chunker(smiles, chunk_size, save_dir)

    for _ in tqdm(map_fn(convert_smiles_chunk_to_mol2, smiles_and_paths)):
        pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to a directory containing .txt files with smiles strings (with a header row)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to a directory where mol2 files will be saved')
    parser.add_argument('--chunk_size', type=int, default=10000,
                        help='Number of molecules saved to each mol2 file')
    parser.add_argument('--max_num_molecules', type=int, default=None,
                        help='Maximum number of molecules to convert')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Whether to compute conformers sequentially rather than in parallel')
    args = parser.parse_args()

    convert_smiles_to_mol2(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        chunk_size=args.chunk_size,
        max_num_molecules=args.max_num_molecules,
        sequential=args.sequential
    )
