from argparse import Namespace
from copy import deepcopy
import csv
import heapq
import json
from logging import Logger
import os
from pprint import pformat
from random import sample
from typing import List, Tuple, Union
import functools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
import pickle
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import GPy

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from .confidence_estimator import confidence_estimator_builder
from .confidence_evaluator import ConfidenceEvaluator
from chemprop.data import StandardScaler, MoleculeDataset
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data, atomistic_to_molecule
from chemprop.models import build_model, train_residual_model
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint, create_logger


def get_atomistic_splits(data_path : str, args: Namespace,
                         logger : Logger = None):
    """
    Grabs a training, validation, and testing set from a data directory.

    :param data_path: Path to the dataset
    :param args: Arguments.
    :param logger: Logger.
    :return: train, val, test sets along with feature and target scalers
    """

    import schnetpack as spk
    from schnetpack import AtomsData
    from schnetpack.datasets import QM9

    # Create the logger
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Get data
    debug('Loading data')

    # Only inner energy at 0 (U0)
    # TODO: Remove download
    qm9data = QM9(data_path, download=True, load_only=[QM9.U0], remove_uncharacterized=True)

    train_size, val_size, test_size = args.split_sizes
    num_train = train_size * len(qm9data)
    num_val = val_size * len(qm9data)
    num_test = test_size * len(qm9data)

    # Data splitting (test is implicitly the left out portion)
    debug(f"Setting schnetpack seed to {args.seed} for data splitting.")
    spk.utils.set_random_seed(args.seed)
    train, val, test = spk.train_test_split(
        data=qm9data,
        num_train=num_train,
        num_val=num_val,
        split_file=os.path.join(args.save_dir, "split.npz")
    )

    # Test contains all remaining data by default, subsample it to only have
    # the desired length
    test_subset_inds = np.random.choice(len(test), int(np.floor(num_test)))
    test = test.create_subset(test_subset_inds)

    train_data, val_data, test_data = atomistic_to_molecule(train), atomistic_to_molecule(val), atomistic_to_molecule(test)

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(qm9data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')


    # Helper function to remove atomrefs from a dataset targets. We can reuse
    # this function for each of the three datasets as a preprocessing step
    # (not currently needed as we only do it on the training set).
    def subtract_atomrefs_from_dataset(dataset):
        targets = dataset.targets()

        # Atomrefs that will need to be subracted from each molecule
        atomrefs = dataset.get_atomref(QM9.U0)[QM9.U0]

        # Loop through each element in the dataset and subtract the atomrefs
        # (and divide by number of atoms in the molecule)
        targets_minus_atomref = []
        num_atoms_list = []
        for example, target in zip(dataset, targets):
            z = example["_atomic_numbers"]
            target -= atomrefs[z].sum()  # subtract

            num_atoms = example['_atom_mask'].sum().item()
            targets_minus_atomref.append(target)#/num_atoms)  # divide
            num_atoms_list.append(num_atoms)

        dataset.set_targets(targets_minus_atomref)
        return dataset, num_atoms_list

    # Now that atomrefs are accounted for, create a mean/std
    # scaler from the training set (and apply)
    train_data, num_atoms_list = subtract_atomrefs_from_dataset(train_data)
    train_targets = train_data.targets()
    scaler = StandardScaler(atomwise=True, no_scale=False).fit(train_targets, num_atoms_list)
    scaled_targets = scaler.transform(train_targets, num_atoms_list).tolist()
    train_data.set_targets(scaled_targets)

    # Include to not scale
    # scaler = None

    return (train_data, val_data, test_data), None, scaler

def get_dataset_splits(data_path: str, args: Namespace, logger: Logger = None):
    """
    Grabs a training, validation, and testing set from a data directory.

    :param data_path: Path to the dataset
    :param args: Arguments.
    :param logger: Logger.
    :return: train, val, test sets along with feature and target scalers
    """

    # Create the logger
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Get data
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    debug(f'Number of tasks = {data.num_tasks}')

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args,
                             features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args,
                            features_path=args.separate_val_features_path, logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(
            0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(
            0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.split_type == "ood_test": 
        ## Export extras for ood testing
        train_data, val_data, test_data, csv_export = split_data(
            data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)
        csv_export_name = os.path.join(args.save_dir, 'ood_info.csv')
        csv_export.to_csv(csv_export_name)
    else:
        train_data, val_data, test_data = split_data(
            data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.save_smiles_splits:
        with open(args.data_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            lines_by_smiles = {}
            indices_by_smiles = {}
            for i, line in enumerate(reader):
                smiles = line[0]
                lines_by_smiles[smiles] = line
                indices_by_smiles[smiles] = i

        all_split_indices = []
        for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['smiles'])
                for smiles in dataset.smiles():
                    writer.writerow([smiles])
            with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for smiles in dataset.smiles():
                    writer.writerow(lines_by_smiles[smiles])
            split_indices = []
            for smiles in dataset.smiles():
                split_indices.append(indices_by_smiles[smiles])
                split_indices = sorted(split_indices)
            all_split_indices.append(split_indices)
        with open(os.path.join(args.save_dir, 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    # Turn this off for atomistic networks
    if args.dataset_type == 'regression' and not args.atomistic:
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    return (train_data, val_data, test_data), features_scaler, scaler

def run_training(train_data: MoleculeDataset, val_data: MoleculeDataset,
                scaler: StandardScaler, features_scaler: StandardScaler,
                args: Namespace, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the
    highest validation score.

    :param train_data: A dataset of training molecules
    :param val_data: A dataset of validation molecules
    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Print args
    debug(pformat(vars(args)))

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)
    args.num_tasks = train_data.num_tasks()


    # Train ensemble of models
    train_func = functools.partial(
        run_training_single_model,
        args, train_data, val_data, scaler, features_scaler, loss_func, metric_func,
    )


    # Add to avoid break on dropout/snapshot
    if args.confidence in ['snapshot', 'dropout']:
        model = train_func(0, return_model = True)
        effective_range = range(1, args.ensemble_size)
    else:
        model = None
        effective_range = range(args.ensemble_size)
    # Fix this in case we need to pass in a model already trained
    train_func = functools.partial(train_func, model=model)

    if args.threads == 1: # run in serial
        for model_idx in effective_range:
            train_func(model_idx, model=model)
    else: # run in parallel
        # Spawn a new process to handle each member of the ensemble
        import multiprocessing, psutil
        multiprocessing.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')
        with multiprocessing.Pool(args.threads) as pool:
            # Kick off the processes
            pool.map(train_func, effective_range)
            # Close when done, this should be taken care of by "with"
            # pool.close()
            # pool.join()

    # List container for all trained models in the ensemble
    ensemble_models = []
    for model_idx in range(args.ensemble_size):
        model = load_checkpoint(os.path.join(
            args.save_dir, f'model_{model_idx}', 'model.pt'),
            cuda=args.cuda, logger=logger, dataset=train_data)
        ensemble_models.append(model)

    return ensemble_models



def run_training_single_model(args, train_data, val_data, scaler,
                              features_scaler, loss_func, metric_func,
                              model_idx, model=None, return_model = False):

    # Tensorboard writer
    print(f"Training model {model_idx}")
    save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
    makedirs(save_dir)
    writer = SummaryWriter(log_dir=save_dir)

    # Model specific logger to avoid locking conflicts with main logger
    logger = create_logger(name='model', save_dir=save_dir, quiet=args.quiet)
    debug, info = logger.debug, logger.info

    # Load/build model
    if args.confidence not in ['snapshot', 'dropout'] or model_idx == 0:
        if args.checkpoint_paths is not None:
            debug(
                f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(
                args.checkpoint_paths[model_idx], current_args=args,
                logger=logger, dataset=train_data
            )
        else:
            debug(f'Building model {model_idx}')
            model = build_model(args, train_data, scaler)

        debug(model)

        debug(f'Number of parameters = {param_count(model):,}')
    if args.cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    # Ensure that model is saved in correct location for evaluation if 0 epochs
    save_checkpoint(os.path.join(save_dir, 'model.pt'),
                    model, scaler, features_scaler, args)
    # save_checkpoint(os.path.join(save_dir, 'model.pt'), model, args)

    # Optimizers
    optimizer = build_optimizer(model, args)

    # Learning rate schedulers
    scheduler = build_lr_scheduler(optimizer, args)

    num_epochs = args.epochs
    if args.confidence == 'snapshot':
        num_epochs = num_epochs // args.ensemble_size

    if args.confidence == 'dropout' and model_idx != 0:
        num_epochs = 0

    # Run training
    best_score = float('inf') if args.minimize_score else -float('inf')
    best_epoch, n_iter = 0, 0
    my_range = range if args.quiet else trange
    for epoch in my_range(num_epochs):
        debug(f'Epoch {epoch}')

        train_data_sample = train_data

        # if args.confidence == 'bootstrap':
        #     print(train_data)
        #     train_data_sample = sample(train_data, int(args.train_data_size * (1.5 / args.ensemble_size)))

        n_iter = train(
            model=model,
            data=train_data_sample,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            n_iter=n_iter,
            logger=logger,
            writer=writer
        )

        if isinstance(scheduler, ExponentialLR):
            scheduler.step()

        val_scores = evaluate(
            model=model,
            data=val_data,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            batch_size=args.batch_size,
            dataset_type=args.dataset_type,
            scaler=scaler,
            logger=logger,
            quiet=args.quiet,
        )

        # Average validation score
        avg_val_score = np.nanmean(val_scores)

        # Step scheduler if atomistic
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_score)

        debug(f'Validation {args.metric} = {avg_val_score:.6f}')
        writer.add_scalar(
            f'validation_{args.metric}', avg_val_score, n_iter)

        if args.show_individual_scores:
            # Individual validation scores
            for task_name, val_score in zip(args.task_names, val_scores):
                debug(
                    f'Validation {task_name} {args.metric} = {val_score:.6f}')
                writer.add_scalar(
                    f'validation_{task_name}_{args.metric}', val_score, n_iter)

        # Save model checkpoint if improved validation score
        if args.minimize_score and avg_val_score < best_score or \
                not args.minimize_score and avg_val_score > best_score:
            if not args.quiet:
                print(f"Saving model, epoch {epoch}, val {avg_val_score:.4f}")
            best_score, best_epoch = avg_val_score, epoch
            save_checkpoint(
                os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

    print(f"Done with model {model_idx}")
    if return_model:
        return model

def evaluate_models(models, train_data, new_data, scaler, args, logger,
                    export_std = False, export_single_model_preds = False):
    """
    Evaluate an ensemble of models on a testing set.

    :param train_data: A dataset of training molecules (used for conf baselines)
    :param new_data: A dataset of molecules to evaluate, *targets must be un-scaled*
    :param scaler: Scaler used to scale targets
    :param args: Arguments.
    :param logger: Logger.
    :param export_std: If true, export std as well (useful for evidence method where we don't export std but only evidence)
    :return: A list of ensemble scores for each task.
    """

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    if args.confidence:
        confidence_estimator = confidence_estimator_builder(args.confidence)(
                                train_data, new_data, scaler, args)

    predictions = []
    for model_idx, model in enumerate(models):
        single_model_preds = predict(
            model=model,
            data=new_data,
            batch_size=args.batch_size,
            scaler=scaler,
            quiet=args.quiet
        )

        single_model_scores = evaluate_predictions(
            preds=single_model_preds,
            targets=new_data.targets(),
            num_tasks=new_data.num_tasks(),
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )

        predictions.append(single_model_preds)

        # Average test score
        avg_single_model_score = np.nanmean(single_model_scores)
        # info(f'Model {model_idx} test {args.metric} = {avg_single_model_score:.6f}')
        # writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)

        if args.confidence:
            confidence_estimator.process_model(model, predict)

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, single_model_scores):
                info(
                    f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
                # writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)

    ensemble_predictions = np.mean(predictions, axis=0)

    # Compute entropy before we modify dropout for inference
    entropy = None
    if args.use_entropy and args.confidence:
        # Convert uncertainties from standard devs to entropy if desired
        if args.dataset_type == 'classification':
            def categorical_entropy(p):
                return -(p*np.log(p) + (1-p)*np.log(1-p))
            entropy = categorical_entropy(ensemble_predictions)

        else:
            def gaussian_entropy(std):
                return -1/2.*np.log(2*np.pi*np.exp(1)*std**2)
            entropy = gaussian_entropy(ensemble_predictions)

    # If we have dropout, correct s.t. the ensemble predictions passed are
    # computed without dropout
    # Note: The predictions in the ensemble used for dropout have already been
    # computed in the confdience_estimator.process_model function, so this
    # shouldn't affect confidence computation!
    if args.confidence == "dropout" and args.no_dropout_inference:
        model.dropout.set_inference_mode(True)
        ensemble_predictions = np.array(predict(
            model=model,
            data=new_data,
            batch_size=args.batch_size,
            scaler=scaler,
            quiet=args.quiet
        ))
        model.dropout.set_inference_mode(False)

    ensemble_scores = evaluate_predictions(
        preds=ensemble_predictions.tolist(),
        targets=new_data.targets(),
        num_tasks=new_data.num_tasks(),
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    # Average ensemble score
    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
    # writer.add_scalar(f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0)

    confidence = None
    std = None
    if args.confidence:
        (ensemble_predictions, confidence) = confidence_estimator.compute_confidence(
                                        ensemble_predictions)

    return_vars = [ensemble_scores, ensemble_predictions, confidence]
    if export_std and args.confidence:
        # Also extract the STD
        std = confidence_estimator.export_std()
        if std is None: std = confidence
        return_vars.append(std)
        # return ensemble_scores, ensemble_predictions, confidence, std, entropy
    # else:
    #     return ensemble_scores, ensemble_predictions, confidence, entropy
    return_vars.append(entropy)
    if export_single_model_preds:
        return_vars.extend([single_model_scores, single_model_preds])

    return return_vars
