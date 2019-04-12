from argparse import ArgumentParser
import csv
import json
from multiprocessing import Pool
from typing import List

from bs4 import BeautifulSoup
import requests
from tqdm import tqdm


def get_smiles_from_id(id: str) -> str:
    try:
        url = f'https://www.ebi.ac.uk/chembl/compound_report_card/{id}'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        json_element = soup.find(id='JSON_LD')
        compound_info = json.loads(json_element.text)
        smiles = compound_info['canonical_smiles'][0]
    except Exception as e:
        print(e)
        print(id)
        return ''

    return smiles


def scrape_chembl_smiles(chembl_id_paths: List[str], chembl_smiles_path: str):
    ids = []
    for chembl_id_path in chembl_id_paths:
        with open(chembl_id_path) as f:
            lines = [line.replace('\0', '') for line in f]  # strip null characters
            ids += [row['ChEMBL ID'] for row in csv.DictReader(lines)]

    print(f'Number of IDs = {len(ids)}')

    smiles = []
    with Pool() as pool:
        for smile in tqdm(pool.imap(get_smiles_from_id, ids), total=len(ids)):
            if smile != '':
                smiles.append(smile)

    print(f'Number of SMILES = {len(smiles)}')
    print(f'Number of invalid IDs = {len(ids) - len(smiles)}')

    with open(chembl_smiles_path, 'w') as f:
        for smile in smiles:
            f.write(smile + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--chembl_id_paths', type=str, nargs='+', required=True,
                        help='Path to .txt with ChEMBL IDs')
    parser.add_argument('--chembl_smiles_path', type=str, required=True,
                        help='Path to a .txt where ChEMBL SMILES will be saved')
    args = parser.parse_args()

    scrape_chembl_smiles(
        chembl_id_paths=args.chembl_id_paths,
        chembl_smiles_path=args.chembl_smiles_path
    )
