from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm

def morgan_fingerprint(smiles: str, radius: int = 3, num_bits: int = 2048, use_counts: bool = False) -> np.ndarray:
    """
    Generates a morgan fingerprint for a smiles string.

    :param smiles: A smiles string for a molecule.
    :param radius: The radius of the fingerprint.
    :param num_bits: The number of bits to use in the fingerprint.
    :param use_counts: Whether to use counts or just a bit vector for the fingerprint
    :return: A 1-D numpy array containing the morgan fingerprint.
    """
    if type(smiles) == str:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles
    if use_counts:
        fp_vect = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    else:
        fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_vect, fp)

    return fp

def vis2d(filepath, num_index, flip_color, save_file, annotate, method, transformer=None):
    train_smiles_inhib = []
    with open(filepath, 'r') as rf:
        rf.readline()
        for line in rf:
            line = line.strip().split(',')
            train_smiles_inhib.append((line[0], float(line[num_index]), morgan_fingerprint(line[0])))

    train_morgans = np.stack([tsi[2] for tsi in train_smiles_inhib])

    if transformer is None:
        if method == 'pca':
            transformer = PCA(n_components=2)
        else:
            transformer = TSNE(n_components=2)
    reduced_train = transformer.fit_transform(train_morgans)

    x = reduced_train[:, 0]
    y = reduced_train[:, 1]
    z = [1-tsi[1] if flip_color else tsi[1] for tsi in train_smiles_inhib]

    with open(save_file, 'w') as wf:
        wf.write('smiles,x,y,color\n')
        for i, tsi in enumerate(train_smiles_inhib):
            wf.write(f'{tsi[0]},{x[i]},{y[i]},{z[i]}\n')

    return transformer

transformer = vis2d('broad/train_2600_dedup.csv', 4, False, 'visualizations/2600_bicarbinhibition_smiles.csv', None, 'tsne')
# vis2d('broad/train_2600_dedup.csv', 2, False, 'visualizations/2600_inhibition_indices.csv', 'indices', 'tsne', transformer)
# vis2d('broad/train_2600_dedup.csv', 2, False, 'visualizations/2600_inhibition_smiles.csv', 'smiles', 'tsne', transformer)
vis2d('bicarbinhibition_10k_test_preds_sorted.csv', 1, True, 'visualizations/10k_bicarbinhibition_smiles.csv', None, 'tsne', transformer)
# vis2d('inhibition_10k_test_preds_sorted.csv', 1, True, 'visualizations/10k_inhibition_indices.csv', 'indices', 'tsne', transformer)
# vis2d('inhibition_10k_test_preds_sorted.csv', 1, True, 'visualizations/10k_inhibition_smiles.csv', 'smiles', 'tsne', transformer)


# filepath = 'broad/train_2600_dedup.csv'
# num_index = 2
# flip_color = False
# # filepath = 'inhibition_10k_test_preds_sorted.csv'
# # num_index = 1
# # flip_color = True
# save_file = '2600_inhibition_indices.svg'
# annotate = 'indices'
# method='tsne'
# vis2d(filepath, num_index, flip_color, save_file, annotate, method)