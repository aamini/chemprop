from argparse import ArgumentParser
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_smiles
from chemprop.utils import makedirs
from .smiles_standardizer import SmilesStandardizer


def find_antibiotics(data_dir: str,
                     antibiotics_path: str,
                     results_path: str):
    paths = [os.path.join(data_dir, path) for path in os.listdir(data_dir) if path.endswith('.txt')]

    antibiotics = set(get_smiles(antibiotics_path))
    standardizer = SmilesStandardizer()
    standardized_antibiotics = {standardizer.standardize(smiles) for smiles in antibiotics}

    makedirs(results_path, isfile=True)

    with open(results_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'file_name',
            'size',
            'exact_matches',
            'exact_matches_percent',
            'standardized_matches',
            'standardize_matches_percent'
        ])

        for path in paths:
            smiles = get_smiles(path)
            count = len(smiles)
            exact_count = sum(1 for smile in smiles if smile in antibiotics)
            standardized_count = sum(1 for smile in smiles if standardizer.standardize(smile) in standardized_antibiotics)
            writer.writerow([
                os.path.basename(path),
                count,
                exact_count,
                exact_count / count * 100,
                standardized_count,
                standardized_count / count * 100
            ])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing .txt files with SMILES strings')
    parser.add_argument('--antibiotics_path', type=str, required=True,
                        help='Path to a .csv file containing antibiotics SMILES strings')
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to a .csv file where the results will be saved')
    args = parser.parse_args()

    find_antibiotics(
        data_dir=args.data_dir,
        antibiotics_path=args.antibiotics_path,
        results_path=args.results_path
    )
