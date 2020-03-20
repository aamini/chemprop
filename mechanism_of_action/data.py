# Collection of methods to pull and process mechanism data.
import csv
from typing import List


class Compound:
    def __init__(self, smile_string: str):
        '''
        :param str smile_string: The smile string of the Compound.
        '''
        self.smile_string = smile_string
        self.chembl_id = self.get_chembl_id()
        self.targets = self.get_targets()

    def get_chembl_id(self) -> str:
        '''
        :returns str: The ChEMBL ID of the Compound.
        '''
        return ""

    def get_targets(self) -> List[Target]:
        '''
        :returns List[Target]: A list of the Compound's targets.
        '''
        return []


class Target:
    def __init__(self, smile_string):
        '''
        :param str smile_string: The smile string of the Target.
        '''
        self.smile_string = smile_string
        self.go_terms = self.get_go_terms()

    def get_go_terms(self) -> List[str]:
        '''
        :returns List[Target]: A list of GO terms associated with the target.
        '''
        return []


def get_targets(source_file: str, output_file: str):
    '''
    Stores a list of targets associated with a set of molecules.

    :param str source_file: A tab-delimited CSV file with a 'SMILES' column.
    :param str output_file: The location targets will be saved.
    :raises ValueError: if the source_file contains no SMILES column.
    '''
    with open(source_file) as compound_file:
        compound_reader = csv.reader(compound_file, delimiter='\t')

        line_count = 0
        smiles_index = 0

        for row in compound_reader:
            if line_count == 0:
                smiles_index = row.index('SMILES')
            else:
                compound = Compound(row[smiles_index])
            line_count += 1
