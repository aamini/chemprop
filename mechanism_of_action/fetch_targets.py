"""Stores a list of targets associated with a set of molecules."""
from chembl_webresource_client.new_client import new_client as chembl_client
import csv
from tap import Tap
from tqdm import tqdm
from typing import Dict, List

from data import Drug


class SimpleArgumentParser(Tap):
    source_file: str  # A tab-delimited CSV file with a 'SMILES' column.
    drug_output_file: str  # The location targets will be saved.
    target_output_file: str  # The location targets will be saved.


def create_drugs(smiles_list: List[str],
                 chunk_size: int = 50) -> Dict[str, Dict]:
    """
    Produce a dictionary of chEMBL IDs to SMILES.

    :param List[str] smiles_list: A list of drug SMILES.
    :param int chunk_size: The number of drug chEMBL IDs to lookup at a time.
    :returns Dict[str, Dict]: A dictionary from drug chEMBL IDs to info.
    """
    drug_ids_to_info = {}

    # Find drug SMILES.
    drugs = []
    missed_count = 0
    for smiles in tqdm(smiles_list, desc='Fetching Drug chEMBL IDs'):
        try:
            drugs.append(Drug(smiles=smiles))
        except ValueError:
            missed_count += 1

    print(f'Could not identify {missed_count}/{len(smiles_list)} drugs.')

    for drug in drugs:
        drug_ids_to_info[drug.chembl_id] = {'smiles': drug.smiles,
                                            'targets': set()}

    # Find drug targets.
    drug_ids = list(drug_ids_to_info.keys())
    for i in tqdm(range(0, len(drug_ids), chunk_size),
                  desc='Fetching Drug Targets'):
        activities = chembl_client.activity.filter(
            molecule_chembl_id__in=drug_ids[i:i+chunk_size]).only([
                    'target_chembl_id', 'molecule_chembl_id'])

        for activity in tqdm(activities, desc='Processing Chunk', leave=False):
            drug_ids_to_info[activity['molecule_chembl_id']]['targets'].add(
                activity['target_chembl_id'])

    return drug_ids_to_info


def write_targets(target_ids: List[str],
                  output_writer: csv.writer,
                  chunk_size: int = 50):
    """
    Write target info using the designated writer.

    :param List[str] target_ids: A list of target chEMBL IDs.
    :param csv.writer output_writer: The CSV writer to use.
    :param int chunk_size: The number of target chEMBL IDs to lookup at a time.
    """
    # Find GO terms associated with each Target.
    for i in tqdm(range(0, len(target_ids), chunk_size),
                  desc='Fetching GO Terms'):
        targets = chembl_client.target.filter(
            target_chembl_id__in=target_ids[i:i+chunk_size]).only([
                'target_chembl_id', 'target_components', 'organism'])

        seen_target_ids = set()
        for target in tqdm(targets, desc='Processing Chunk', leave=False):
            if target['target_chembl_id'] in seen_target_ids:
                continue

            seen_target_ids.add(target['target_chembl_id'])
            go_terms = {'GoComponent': set(),
                        'GoFunction': set(),
                        'GoProcess': set()}

            for component in target['target_components']:
                xrefs = component['target_component_xrefs']
                for xref in xrefs:
                    if xref['xref_src_db'] not in ['GoComponent',
                                                   'GoFunction',
                                                   'GoProcess']:
                        continue

                    go_terms[xref['xref_src_db']].add(xref['xref_id'])

                output_writer.writerow([target['target_chembl_id'],
                                        target['organism'],
                                        ",".join(go_terms['GoComponent']),
                                        ",".join(go_terms['GoFunction']),
                                        ",".join(go_terms['GoProcess'])])


if __name__ == '__main__':
    args = SimpleArgumentParser().parse_args()

    smiles_list = []
    with open(args.source_file) as compound_file:
        compound_reader = csv.reader(compound_file, delimiter='\t')

        line_count = 0
        smiles_index = 0

        for row in compound_reader:
            if line_count == 0:
                smiles_index = row.index('SMILES')
            else:
                smiles_list += row[smiles_index].split(', ')

            line_count += 1

    drug_ids_to_info = create_drugs(smiles_list)

    with open(args.drug_output_file, 'w+') as output_file:
        output_writer = csv.writer(output_file, delimiter='\t')
        output_writer.writerow(['SMILES', 'chEMBL ID', 'Target chEMBL IDs'])

        for chembl_id, info in drug_ids_to_info.items():
            output_writer.writerow([info['smiles'],
                                    chembl_id,
                                    ",".join(info['targets'])])

    target_ids = list(set.union(
        *[drug['targets'] for drug in drug_ids_to_info.values()]))

    with open(args.target_output_file, 'w+') as output_file:
        output_writer = csv.writer(output_file, delimiter='\t')
        output_writer.writerow(['chEMBL ID',
                                'organism',
                                'GOComponent Terms',
                                'GoFunction Terms',
                                'GoProcess Terms'])

        write_targets(target_ids, output_writer)
