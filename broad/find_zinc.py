from argparse import ArgumentParser
import csv
import json
from multiprocessing import Pool
import os
from tqdm import tqdm

from standardize_smiles import standardize_smiles


def find_zinc(missing_path: str, zinc_dir: str, save_path: str):
    missing = {}
    standard_to_original = {}

    with open(missing_path) as f:
        for row in csv.DictReader(f):
            if row['zinc_index'].strip() != '':
                continue

            smiles = row['smiles']
            standard_smiles = standardize_smiles(smiles)
            standard_to_original[standard_smiles] = smiles

            missing[smiles] = {
                'original': [],
                'standard': []
            }

    print(f'{len(missing)} missing zinc indices')

    paths = sorted([os.path.join(zinc_dir, fname) for fname in os.listdir(zinc_dir) if fname.endswith('.txt')])

    for path in paths:
        tranche = os.path.splitext(os.path.basename(path))[0]
        print(tranche)

        with open(path) as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
            smiles_list = [row['smiles'] for row in rows]

            with Pool() as pool:
                standard = list(tqdm(pool.imap(standardize_smiles, smiles_list), total=len(smiles_list)))

            for row, standard_smiles in tqdm(zip(rows, standard), total=len(rows)):
                smiles = row['smiles']

                zinc_row = {
                    'smiles': smiles,
                    'zinc_index': row['zinc_id'],
                    'tranche': tranche
                }

                if smiles in missing:
                    missing[smiles]['original'].append(zinc_row)

                if standard_smiles in standard_to_original:
                    missing[standard_to_original[standard_smiles]]['standard'].append(zinc_row)

    with open(save_path, 'w') as f:
        json.dump(missing, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--missing_path', type=str, required=True,
                        help='Path to csv file with missing zinc ids')
    parser.add_argument('--zinc_dir', type=str, required=True,
                        help='Path to directory containing zinc files')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to json file where results will be saved')
    args = parser.parse_args()

    find_zinc(
        missing_path=args.missing_path,
        zinc_dir=args.zinc_dir,
        save_path=args.save_path
    )
