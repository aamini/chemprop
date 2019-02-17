from molvs import Standardizer
from rdkit import Chem


class SmilesStandardizer:
    def __init__(self):
        self.mol_standardizer = Standardizer()

    def standardize(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol = self.mol_standardizer.standardize(mol)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

        return smiles
