from argparse import ArgumentParser
import csv
from multiprocessing import Pool
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tqdm import tqdm

from chemprop.data.utils import get_smiles
from chemprop.utils import makedirs


def find_antibiotics(data_dir: str,
                     antibiotics_path: str,
                     results_path: str,
                     standardize: bool,
                     restart: bool):
    if standardize:
        # from molvs import standardize_smiles
        from standardize_smiles import standardize_smiles

    # Load partially complete results if not restarting
    if not restart and os.path.exists(results_path):
        with open(results_path) as f:
            reader = csv.DictReader(f)
            completed_file_names = {row['file_name'] for row in reader}
        results_path_exists = True
    else:
        completed_file_names = set()
        results_path_exists = False

    # Load data paths
    paths = [os.path.join(data_dir, path) for path in sorted(os.listdir(data_dir)) if path.endswith('.txt') and os.path.basename(path) not in completed_file_names]

    # Load antibiotics
    antibiotics = set(get_smiles(antibiotics_path))
    if standardize:
        standardized_antibiotics = {standardize_smiles(smiles) for smiles in antibiotics}

    # Make directory for results
    makedirs(results_path, isfile=True)

    # Search for antibiotics and store results
    with open(results_path, 'a+') as f:
        writer = csv.writer(f)

        # Write header if first creating file
        if not results_path_exists:
            header = [
                'file_name',
                'size',
                'exact_matches',
                'exact_matches_percent'
            ]
            if standardize:
                header += [
                    'standardized_matches',
                    'standardized_matches_percent'
                ]
            writer.writerow(header)

        # Find antibiotics and write results
        for path in tqdm(paths, total=len(paths)):
            smiles = get_smiles(path)
            count = len(smiles)
            exact_count = sum(1 for smile in tqdm(smiles, total=count) if smile in antibiotics)
            row = [
                os.path.basename(path),
                count,
                exact_count,
                exact_count / count * 100
            ]

            if standardize:
                with Pool() as pool:
                    standardized_count = sum(1 for smile in tqdm(pool.imap_unordered(standardize_smiles, smiles), total=count) if smile in standardized_antibiotics)
                row += [
                    standardized_count,
                    standardized_count / count * 100
                ]

            writer.writerow(row)
            f.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing .txt files with SMILES strings')
    parser.add_argument('--antibiotics_path', type=str, required=True,
                        help='Path to a .csv file containing antibiotics SMILES strings')
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to a .csv file where the results will be saved')
    parser.add_argument('--standardize', action='store_true', default=False,
                        help='Whether to standardize SMILES strings')
    parser.add_argument('--restart', action='store_true',default=False,
                        help='Whether to restart rather than loading temporary progress')
    args = parser.parse_args()

    find_antibiotics(
        data_dir=args.data_dir,
        antibiotics_path=args.antibiotics_path,
        results_path=args.results_path,
        standardize=args.standardize,
        restart=args.restart
    )
