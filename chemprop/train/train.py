from argparse import Namespace
from copy import deepcopy
import logging
from typing import Callable, List, Union

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from chemprop.data import MoleculeDataset, get_data_batches
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

    data = deepcopy(data)

    if args.confidence == 'bootstrap':
        data.sample(int(4 * len(data) / args.ensemble_size))

    loss_sum, iter_count = 0, 0

    iter_size = args.batch_size

    for index, (batch, features_batch, target_batch, mol_batch_len) in \
            enumerate(get_data_batches(data, iter_size, use_last=False,
                                       shuffle=True, quiet=args.quiet)):

        # Prepare batch
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()

        class_weights = torch.ones(targets.shape)

        if args.cuda:
            class_weights = class_weights.cuda()

        # Run model
        model.zero_grad()
        preds = model(batch, features_batch)

        if model.confidence: # confidence is learned
            if args.confidence == 'evidence':
                if args.dataset_type == 'regression': # normal inverse gamma
                    # split into four parameters and feed into loss
                    #means, lambdas, alphas, betas = torch.split(preds, preds.shape[1]//4, dim=1)
                    means =  preds[:, [j for j in range(len(preds[0])) if j % 4 == 0]]
                    lambdas =  preds[:, [j for j in range(len(preds[0])) if j % 4 == 1]]
                    alphas =  preds[:, [j for j in range(len(preds[0])) if j % 4 == 2]]
                    betas  =  preds[:, [j for j in range(len(preds[0])) if j % 4 == 3]]
                    loss = loss_func(means, lambdas, alphas, betas, targets)
                if args.dataset_type == 'classification': # dirichlet
                    loss = loss_func(targets, alphas=preds)
            else: # gaussian MVE for regression
                assert args.dataset_type == 'regression'
                pred_targets = preds[:, [j for j in range(len(preds[0])) if j % 2 == 0]]
                pred_var = preds[:, [j for j in range(len(preds[0])) if j % 2 == 1]]
                loss = loss_func(pred_targets, pred_var, targets)

        else:
            loss = loss_func(preds, targets)

        # Only apply loss to valid tasks targets and then average over samples
        loss = loss * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += mol_batch_len

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += mol_batch_len

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

        # Debug condition
        if index > 2 and args.debug:
            break

    return n_iter
