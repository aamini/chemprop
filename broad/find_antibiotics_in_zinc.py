from argparse import ArgumentParser
import csv
from multiprocessing import Pool
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tqdm import tqdm

from chemprop.data.utils import get_smiles
from chemprop.utils import makedirs


def find_antibiotics_in_zinc(data_dir: str,
                             antibiotics_path: str,
                             save_dir: str,
                             standardize: bool,
                             restart: bool):
    if standardize:
        # from molvs import standardize_smiles
        from standardize_smiles import standardize_smiles

    # Create directory for results and specify filenames
    makedirs(save_dir)
    stats_save_path = os.path.join(save_dir, 'stats.csv')
    exact_matches_path = os.path.join(save_dir, 'exact_matches.csv')
    standardized_matches_path = os.path.join(save_dir, 'standardized_matches.csv')

    # Load partially complete results if not restarting
    if not restart and os.path.exists(stats_save_path):
        with open(stats_save_path) as f:
            reader = csv.DictReader(f)
            completed_file_names = {row['file_name'] for row in reader}
        results_exist = True
    else:
        completed_file_names = set()
        results_exist = False

    # Load data paths
    paths = [os.path.join(data_dir, fname) for fname in sorted(os.listdir(data_dir))
             if fname.endswith('.txt') and fname not in completed_file_names]

    # Load antibiotics
    antibiotics = set(get_smiles(antibiotics_path))
    if standardize:
        standardized_antibiotics = {standardize_smiles(smiles) for smiles in antibiotics}

    # Search for antibiotics and store results
    with open(stats_save_path, 'a+') as f_stats,\
            open(exact_matches_path, 'a+') as f_exact,\
            open(standardized_matches_path, 'a+') as f_standardized:
        stats_writer, exact_writer, standardized_writer = csv.writer(f_stats), csv.writer(f_exact), csv.writer(f_standardized)

        # Write header if first creating file
        if not results_exist:
            header = [
                'chunk',
                'size',
                'exact_matches',
                'exact_matches_percent'
            ]
            if standardize:
                header += [
                    'standardized_matches',
                    'standardized_matches_percent'
                ]
            stats_writer.writerow(header)

        # Find antibiotics and write results
        for path in tqdm(paths, total=len(paths)):
            chunk = os.path.splitext(os.path.basename(path))[0]
            smiles = get_smiles(path)
            count = len(smiles)
            exact_matches = [smile for smile in tqdm(smiles, total=count) if smile in antibiotics]
            exact_count = len(exact_matches)
            row = [
                chunk,
                count,
                exact_count,
                exact_count / count * 100
            ]

            if standardize:
                with Pool() as pool:
                    standardized_matches = [smile for smile in tqdm(pool.imap_unordered(standardize_smiles, smiles), total=count) if smile in standardized_antibiotics]
                    standardized_count = len(standardized_matches)
                row += [
                    standardized_count,
                    standardized_count / count * 100
                ]

            stats_writer.writerow(row)
            exact_writer.writerow([chunk] + sorted(list(set(exact_matches))))
            if standardize:
                standardized_writer.writerow([chunk] + sorted(list(set(standardized_matches))))

            f_stats.flush()
            f_exact.flush()
            if standardize:
                f_standardized.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing .txt files with SMILES strings')
    parser.add_argument('--antibiotics_path', type=str, required=True,
                        help='Path to a .csv file containing antibiotics SMILES strings')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to a directory where the results will be saved')
    parser.add_argument('--standardize', action='store_true', default=False,
                        help='Whether to standardize SMILES strings')
    parser.add_argument('--restart', action='store_true',default=False,
                        help='Whether to restart rather than loading temporary progress')
    args = parser.parse_args()

    find_antibiotics_in_zinc(
        data_dir=args.data_dir,
        antibiotics_path=args.antibiotics_path,
        save_dir=args.save_dir,
        standardize=args.standardize,
        restart=args.restart
    )
