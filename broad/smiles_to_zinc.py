from argparse import ArgumentParser
import csv
import json
from multiprocessing import Pool
from typing import List, Tuple

import requests
from tqdm import tqdm


def smiles_to_zinc(smiles: str, count: int = 5) -> Tuple[str, List[Tuple[str, str]]]:
    # Find the zinc id corresponding to a smiles string
    res = requests.get(f'http://zinc15.docking.org/substances.txt:smiles+zinc_id?structure-contains={smiles}&count={count}')
    found_smiles_and_zinc = [line.split('\t') for line in res.text.split('\n')]  # [(smiles, zinc_id), (smiles, zinc_id), ...]

    return smiles, found_smiles_and_zinc


def missing_smiles_to_zinc(missing_path: str, save_path: str):
    # Determine smiles that are missing zinc ids
    with open(missing_path) as f:
        missing_smiles = [row['smiles'] for row in csv.DictReader(f) if row['zinc_index'].strip() == '']

    print(f'There are {len(missing_smiles)} smiles missing zinc ids')

    # Get the zinc ids corresponding to the smiles
    with Pool() as pool:
        smiles_to_zinc_mapping = dict(tqdm(pool.imap(smiles_to_zinc, missing_smiles), total=len(missing_smiles)))

    # Save the smiles and zinc ids
    with open(save_path, 'w') as f:
        json.dump(smiles_to_zinc_mapping, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--missing_path', type=str, default='/data/rsg/chemistry/swansonk/chemprop/broad/zinc_predictions.csv',
                        help='Path to csv file with missing zinc ids')
    parser.add_argument('--save_path', type=str, default='/data/rsg/chemistry/swansonk/chemprop/broad/zinc_predictions_missing_from_url.json',
                        help='Path to json file where results will be saved')
    args = parser.parse_args()

    missing_smiles_to_zinc(
        missing_path=args.missing_path,
        save_path=args.save_path
    )
