from argparse import Namespace
from typing import List

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from chemprop.data import MoleculeDataset, StandardScaler


def predict(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None,
            args: Namespace = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :param args: Arguments.
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

        # If pretraining, select substructure/context pairs
        if args.dataset_type == 'pretraining':
            # Extract substructure vecs and molecule ids
            substructure_vecs = batch_preds[torch.arange(0, len(batch_preds), 2)]

            # Extract context vecs and molecule ids
            context_vecs = batch_preds[torch.arange(1, len(batch_preds), 2)]

            # Get molecule index for each substructure/context
            mol_scope = mol_batch.mol_scope()

            # Check lengths
            assert len(mol_scope) == len(substructure_vecs) == len(context_vecs)

            # Sample negative substructure/context pairs
            substructure_vecs = substructure_vecs.repeat(1 + args.num_negatives_per_positive, 1)

            for _ in range(args.num_negatives_per_positive):
                for mol_index in mol_scope:
                    negative_indices = np.where(mol_scope != mol_index)[0]
                    negative_index = np.random.choice(negative_indices)
                    context_vecs = torch.cat((context_vecs, context_vecs[negative_index].unsqueeze(dim=0)), dim=0)

            # Dot product comparison
            batch_preds = torch.bmm(substructure_vecs.unsqueeze(dim=1), context_vecs.unsqueeze(dim=2)).squeeze(dim=2)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
