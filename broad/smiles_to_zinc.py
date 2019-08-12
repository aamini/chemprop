from argparse import ArgumentParser
import csv

import requests


def smiles_to_zinc(smiles: str) -> str:
    # Find the zinc id corresponding to a smiles string
    res = requests.get(f'http://zinc15.docking.org/substances.txt:smiles+zinc_id?structure-contains={smiles}&count=all')
    # TODO: extract the smiles and zinc ids and get the zinc id corresponding to the result which exactly matches the provided smiles string
    text = res.text

    return text


def missing_smiles_to_zinc(missing_path: str, save_path: str):
    # Determine smiles that are missing zinc ids
    with open(missing_path) as f:
        missing_smiles = [row['smiles'] for row in csv.DictReader(f) if row['zinc_index'].strip() == '']

    # Get the zinc ids corresponding to the smiles
    zinc_ids = [smiles_to_zinc(smiles) for smiles in missing_smiles]

    # Save the smiles and zinc ids
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles', 'zinc_id'])

        for smiles, zinc_id in zip(missing_smiles, zinc_ids):
            writer.writerow([smiles, zinc_id])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--missing_path', type=str, default='/data/rsg/chemistry/swansonk/chemprop/broad/zinc_predictions.csv',
                        help='Path to csv file with missing zinc ids')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to json file where results will be saved')
    args = parser.parse_args()

    missing_smiles_to_zinc(
        missing_path=args.missing_path,
        save_path=args.save_path
    )
