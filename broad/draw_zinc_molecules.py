from argparse import ArgumentParser
import csv
import os
import sys

from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_smiles
from chemprop.utils import makedirs


def draw_molecules(data_path: str, save_dir: str):
    # Load data
    with open(data_path) as f:
        data = list(csv.DictReader(f))

    # Save images
    makedirs(save_dir)
    for d in tqdm(data, total=len(data)):
        mol = Chem.MolFromSmiles(d['smiles'])
        Draw.MolToFile(mol, os.path.join(save_dir, f'{d["zinc_index"]}.png'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file containing SMILES strings and zinc indices')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to directory where PNG images of molecules will be saved')
    args = parser.parse_args()

    draw_molecules(args.data_path, args.save_dir)
