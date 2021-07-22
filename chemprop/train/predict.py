from typing import List

import numpy as np
import torch
import torch.nn as nn

from chemprop.data import MoleculeDataset, StandardScaler, AtomisticDataset, get_data_batches


def predict(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None,
            confidence: bool = False,
            quiet : bool = False,
            export_var: bool = False) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :param confidence: Whether confidence values should be returned.
    :param export_var: If true, export var in addition to confidence (useful
        for evidence)
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []

    iter_size = batch_size


    for batch, features_batch, target_batch, mol_batch_len in \
            get_data_batches(data, iter_size, use_last=True, shuffle=False, quiet=quiet):

        with torch.no_grad():
            batch_preds = model(batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    if model.confidence:
        p = []
        c = []
        var = []
        for i in range(len(preds)):
            if model.conf_type == "nn":
                p.append([preds[i][j] for j in range(len(preds[i])) if j % 2 == 0])
                c_vals = [preds[i][j] for j in range(len(preds[i])) if j % 2 == 1]
                c.append(c_vals)
                var.append(c_vals)
            elif model.conf_type == "evidence":
                # Classification
                if model.classification:
                    # Hard code to 2 classes per task, since this assumption is already made
                    # for the existing chemprop classification tasks
                    num_classes = 2

                    alphas = preds[i] #shape=(num_tasks * num_classes)
                    num_tasks = len(alphas)//num_classes

                    alphas = np.reshape(alphas, (num_tasks, num_classes))
                    evidence = alphas - 1
                    probs = alphas / np.sum(alphas, axis=-1).reshape(num_tasks, 1)

                    # final probability is just the prob of being active in
                    # this task
                    probs = probs[:,1]
                    p.append(probs)

                    conf = num_classes / np.sum(alphas, axis=-1)
                    c.append(conf)
                    # TODO: std not implemented here
                    var.append(conf)

                # Regression
                else:
                    # Switching to chemprop implementation
                    means = np.array([preds[i][j] for j in range(len(preds[i])) if j % 4== 0])
                    lambdas = np.array([preds[i][j] for j in range(len(preds[i])) if j % 4== 1])
                    alphas =  np.array([preds[i][j] for j in range(len(preds[i])) if j % 4== 2])
                    betas =  np.array([preds[i][j] for j in range(len(preds[i])) if j % 4== 3])
                    #means, lambdas, alphas, betas = np.split(np.array(preds[i]), 4)

                    inverse_evidence = 1. / ((alphas-1) * lambdas)

                    p.append(means)

                    #NOTE: inverse-evidence (ie. 1/evidence) is also a measure of
                    # confidence. we can use this or the Var[X] defined by NIG.
                    c.append(inverse_evidence)
                    var.append(betas * inverse_evidence)
                    # c.append(betas / ((alphas-1) * lambdas))
            else:
                raise Exception(f"Conf type {model.conf_type} is not recognized")

        if scaler is not None:


            num_atoms_list = None
            if type(data) is AtomisticDataset and scaler.atomwise: 
                num_atoms_list = get_num_atoms_list(data)

            p = scaler.inverse_transform(p, num_atoms_list).tolist()

            #p = scaler.inverse_transform(p).tolist()

            # Need to add back in the atomrefs now to bring back to the scale
            # of data if we are using an atomistic network
            if type(data) is AtomisticDataset:
                p = inverse_transform_atomrefs(data, p)

            # These are BOTH variances
            c = (scaler.stds**2 * c).tolist()
            var = (scaler.stds**2 * var).tolist()

        if confidence and export_var:
            return p, c, var
        elif confidence:
            return p, c

        return p

    if scaler is not None:
        # Inverse transform now takes a list of atoms if it's atomistic 
        num_atoms_list = None
        if type(data) is AtomisticDataset and scaler.atomwise: 
            num_atoms_list = get_num_atoms_list(data)

        preds = scaler.inverse_transform(preds, num_atoms_list).tolist()

        # Need to add back in the atomrefs now to bring back to the scale
        # of data if we are using an atomistic network
        if type(data) is AtomisticDataset:
            preds = inverse_transform_atomrefs(data, preds)

    return preds

def get_num_atoms_list(data): 
    """ get nm atoms """
    return [example['_atom_mask'].sum().item() for example in data]


def inverse_transform_atomrefs(data, preds):
    """ For atomistic networks we need to re-add the atomrefs to invert the
    original transformation. Call this method with the input dataset and
    transformed predictions, the method will return the inverse transformed
    targets such that they are in the same units as data.targets()
    """

    from schnetpack.datasets import QM9

    # Get atomrefs and add back in one example at a time
    atomrefs = data.get_atomref(QM9.U0)[QM9.U0]
    preds_plus_atomref = []
    for example, pred in zip(data, preds):
        num_atoms = example['_atom_mask'].sum().item()
        z = example["_atomic_numbers"]

        # Apply inverse transformation
        #pred = num_atoms * np.array(pred) + atomrefs[z].sum()
        pred = np.array(pred) + atomrefs[z].sum()
        preds_plus_atomref.append(list(pred))

    # Save the inverse scaled predictions
    preds = preds_plus_atomref
    return preds
