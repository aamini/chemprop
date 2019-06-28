from argparse import Namespace
import logging
from typing import Callable, List, Union

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange

from chemprop.data import MoleculeDataset
from chemprop.features import mol2graph
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model: nn.Module,
          data: Union[MoleculeDataset, List[MoleculeDataset]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print
    
    model.train()
    
    data.shuffle()

    loss_sum, iter_count = 0, 0

    num_iters = len(data) // args.batch_size * args.batch_size  # don't use the last batch if it's small, for stability

    iter_size = args.batch_size

    for i in trange(0, num_iters, iter_size):
        # Prepare batch
        if i + args.batch_size > len(data):
            break
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = smiles_batch
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        # If pretraining, convert smiles to BatchMolGraph before model
        if args.dataset_type == 'pretraining':
            # Convert smiles to subgraphs
            batch = mol2graph(batch, args)

            # Extract molecule scope
            mol_scope = np.array([batch.mol_scope[i] for i in range(0, len(batch.mol_scope), 2)])

            # Define targets
            targets = torch.FloatTensor([[1]] * len(mol_scope) + [[0]] * len(mol_scope) * args.num_negatives_per_positive)

            # Define mask
            mask = torch.ones(targets.shape)

        class_weights = torch.ones(targets.shape)

        if next(model.parameters()).is_cuda:
            mask, targets, class_weights = mask.cuda(), targets.cuda(), class_weights.cuda()

        # Run model
        model.zero_grad()
        preds = model(batch, features_batch)

        # If pretraining, select substructure/context pairs
        if args.dataset_type == 'pretraining':
            # Extract substructure vecs and molecule ids
            substructure_vecs = preds[torch.arange(0, len(preds), 2)]

            # Extract context vecs and molecule ids
            context_vecs = preds[torch.arange(1, len(preds), 2)]

            # Check lengths
            assert len(mol_scope) == len(substructure_vecs) == len(context_vecs)

            if args.cuda:
                targets = targets.cuda()

            # Sample negative substructure/context pairs
            substructure_vecs = substructure_vecs.repeat(1 + args.num_negatives_per_positive, 1)

            for _ in range(args.num_negatives_per_positive):
                for mol_index in mol_scope:
                    negative_indices = np.where(mol_scope != mol_index)[0]
                    negative_index = np.random.choice(negative_indices)
                    context_vecs = torch.cat((context_vecs, context_vecs[negative_index].unsqueeze(dim=0)), dim=0)

            # Dot product comparison
            preds = torch.bmm(substructure_vecs.unsqueeze(dim=1), context_vecs.unsqueeze(dim=2)).squeeze(dim=2)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds, targets) * class_weights * mask

        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += len(mol_batch)

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(mol_batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter
