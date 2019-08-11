from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

remover = SaltRemover()


def standardize_smiles(smiles: str) -> str:
    smiles = smiles.replace('\\', '')
    smiles = smiles.replace('/', '')
    smiles = smiles.replace('@', '')
    mol = Chem.MolFromSmiles(smiles)
    res = remover.StripMol(mol, dontRemoveEverything=True)
    smiles = Chem.MolToSmiles(res)

    return smiles
