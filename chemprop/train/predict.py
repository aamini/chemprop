from typing import List

import torch
import torch.nn as nn
from tqdm import trange

from chemprop.data import MoleculeDataset, StandardScaler


def predict(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None,
            confidence: bool = False) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :param confidence: Whether confidence values should be returned.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []

    num_iters, iter_step = len(data), batch_size

    for i in trange(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        with torch.no_grad():
            batch_preds = model(batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    if model.confidence:
        p = []
        c = []
        for i in range(len(preds)):
            p.append([preds[i][j] for j in range(len(preds[i])) if j % 2 == 0])
            c.append([preds[i][j] for j in range(len(preds[i])) if j % 2 == 1])

        if scaler is not None:
            p = scaler.inverse_transform(p).tolist()
            c = (scaler.stds**2 * c).tolist()

        if confidence:
            return p, c

        return p

    if scaler is not None:
        preds = scaler.inverse_transform(preds).tolist()

    return preds
