from argparse import Namespace
from typing import Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.spatial.distance import cosine
import torch

from chemprop.models import MoleculeModel
from chemprop.nn_utils import compute_molecule_vectors
from .data import MoleculeDataset


def adjacency_matrix(data: MoleculeDataset,
                     model: MoleculeModel,
                     method: str,
                     batch_size: int) -> np.ndarray:
    matrix = np.zeros((len(data), len(data)))  # diagonal entries should be 0

    if method == 'embedding':
        vecs = compute_molecule_vectors(model=model, data=data, batch_size=batch_size)
        for i, vec_1 in enumerate(vecs):
            for j, vec_2 in enumerate(vecs[i + 1:], i + 1):
                distance = cosine(vec_1, vec_2)
                matrix[i][j] = distance
                matrix[j][i] = distance
    elif method == 'tanimoto':
        fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smiles), 3) for smiles in data.smiles()]
        for i, fp_1 in enumerate(fps):
            for j, fp_2 in enumerate(fps[i + 1:], i + 1):
                distance = 1 - DataStructs.TanimotoSimilarity(fp_1, fp_2)
                matrix[i][j] = distance
                matrix[j][i] = distance
    else:
        raise ValueError(f'Adjacency method "{method}" is not available.')

    return matrix


# following https://arxiv.org/pdf/1904.04717.pdf
def propagate_labels(model: MoleculeModel,
                     data: MoleculeDataset,
                     transductive_data: MoleculeDataset,
                     similarity_model: MoleculeModel,
                     args: Namespace) -> Tuple[MoleculeDataset, torch.FloatTensor]:
    # copy probably unneeded; remove if we need speed
    # data = deepcopy(data)
    # transductive_data = deepcopy(transductive_data)
    # optimize speed with tricks in the future if needed; see paper
    assert args.dataset_type == 'classification'  # in theory could support multiclass too
    assert args.num_tasks == 1  # just one task for now
    assert 0 <= args.label_prop_alpha < 1

    new_data = MoleculeDataset(data.data + transductive_data.data)
    labeled_data_size = len(data)
    data_size = len(new_data)
    W = adjacency_matrix(
        data=new_data,
        model=similarity_model,
        method=args.adjacency_method,
        batch_size=args.batch_size
    )  # TODO add to parsing
    D_neghalf = np.diag(1 / np.sqrt(W.sum(axis=0)))  # D = numpy.diag(W.sum(axis=0))
    W_norm = np.matmul(np.matmul(D_neghalf, W), D_neghalf)
    Y = np.zeros((data_size, 2))  # should be 0 for all unlabeled
    for i, label in enumerate(data.targets()):
        Y[i, int(label[0])] = 1
    Z = np.matmul(np.linalg.inv(np.identity(data_size) - args.label_prop_alpha * W_norm), Y)
    hard_labels = np.argmax(Z, axis=1)
    hard_labels[:labeled_data_size] = [t[0] for t in data.targets()]
    new_data.set_targets([[l] for l in hard_labels])
    Z_norm = Z / Z.sum(axis=1, keepdims=True)
    Z_entropy = np.sum(-Z_norm * np.log(Z_norm), axis=1)
    weights = 1 - Z_entropy / np.log(2)
    weights[:labeled_data_size] = 1  # we're sure about these labels
    # try further class balancing if we need it?

    return new_data, torch.from_numpy(weights).float()
