from argparse import Namespace
import csv
import heapq
import json
from logging import Logger
import os
from pprint import pformat
from typing import List, Tuple, Union

import forestci as fci
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
import pickle
from torch.optim.lr_scheduler import ExponentialLR
import GPy

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from .confidence import confidence_estimator_builder
from chemprop.data import StandardScaler
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import build_model, train_residual_model
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint


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
        fp_vect = AllChem.GetHashedMorganFingerprint(
            mol, radius, nBits=num_bits)
    else:
        fp_vect = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=num_bits)
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_vect, fp)

    return fp


def run_training(args: Namespace, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

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

    # Get data
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')

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
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))
    all_test_preds = np.zeros(
        (len(test_smiles), args.num_tasks, args.ensemble_size))

    if args.confidence:
        confidence_estimator = confidence_estimator_builder(args.confidence)

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        writer = SummaryWriter(log_dir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug(
                f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(
                args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'),
                        model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                data=train_data,
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
                logger=logger
            )

            # Average validation score
            avg_val_score = np.nanmean(val_scores)
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
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'),
                                model, scaler, features_scaler, args)

        # Evaluate on test set using model with best validation score
        info(
            f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(
            save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

        test_preds = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler,
        )

        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)
            all_test_preds[:, :, model_idx] = np.array(test_preds)

        # Average test score
        avg_test_score = np.nanmean(test_scores)
        info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
        writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)

        if args.confidence:
            confidence_estimator.process_model(model, predict, batch_size, scaler)

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(
                    f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
                writer.add_scalar(
                    f'test_{task_name}_{args.metric}', test_score, n_iter)

    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / args.ensemble_size)

    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds.tolist(),
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    # Average ensemble score
    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
    writer.add_scalar(
        f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0)

    targets = np.array(test_targets)
    if args.confidence and args.dataset_type == 'regression':
        if args.confidence == 'gaussian':
            # TODO: Scale confidence to reflect scaled predictions.
            predictions = np.ndarray(
                shape=(len(test_smiles), args.num_tasks))
            confidence = np.ndarray(
                shape=(len(test_smiles), args.num_tasks))

            for task in range(args.num_tasks):
                kernel = GPy.kern.Linear(input_dim=args.last_hidden_size)
                gaussian = GPy.models.SparseGPRegression(
                    avg_last_hidden, transformed_val[:, task:task+1], kernel)
                gaussian.optimize()

                avg_test_preds, avg_test_var = gaussian.predict(
                    avg_last_hidden_test)

                # Scale Data
                domain = np.max(avg_test_var) - np.min(avg_test_var)
                # Shift.
                avg_test_var = avg_test_var - np.min(avg_test_var)
                # Scale domain to 1.
                avg_test_var = avg_test_var / domain
                # Apply log scale and flip.
                avg_test_var = np.maximum(
                    0, -np.log(avg_test_var + np.exp(-10)))

                predictions[:, task:task+1] = avg_test_preds
                confidence[:, task:task+1] = avg_test_var

            predictions = scaler.inverse_transform(predictions)
        elif args.confidence == 'random_forest':
            # TODO: Scale confidence to reflect scaled predictions.
            predictions = np.ndarray(
                shape=(len(test_smiles), args.num_tasks))
            confidence = np.ndarray(
                shape=(len(test_smiles), args.num_tasks))

            n_trees = 100
            for task in range(args.num_tasks):
                forest = RandomForestRegressor(n_estimators=n_trees)
                forest.fit(avg_last_hidden, transformed_val[:, task])

                avg_test_preds = forest.predict(avg_last_hidden_test)
                predictions[:, task] = avg_test_preds

                avg_test_var = fci.random_forest_error(
                    forest, avg_last_hidden, avg_last_hidden_test) * (-1)
                confidence[:, task] = avg_test_var

            predictions = scaler.inverse_transform(predictions)
        elif args.confidence == 'tanimoto':
            predictions = avg_test_preds
            confidence = np.ndarray(
                shape=(len(test_smiles), args.num_tasks))

            train_smiles_sfp = [morgan_fingerprint(s) for s in train_smiles]
            for i in range(len(test_smiles)):
                confidence[i, :] = np.ones((args.num_tasks)) * tanimoto(test_smiles[i], lambda x: max(x))
        elif args.confidence == 'ensemble':
            predictions = avg_test_preds
            confidence = np.var(all_test_preds, axis=2) * -1
        elif args.confidence == 'nn':
            predictions = avg_test_preds
            confidence = sum_test_confidence / args.ensemble_size * (-1)


    if args.confidence and args.dataset_type == 'classification':
        if args.confidence == 'gaussian':
            predictions = np.ndarray(
                shape=(len(test_smiles), args.num_tasks))
            confidence = np.ndarray(
                shape=(len(test_smiles), args.num_tasks))

            val_targets = np.array(val_data.targets())

            for task in range(args.num_tasks):
                kernel = GPy.kern.Linear(input_dim=args.last_hidden_size)

                mask = val_targets[:, task] != None
                gaussian = GPy.models.GPClassification(
                    avg_last_hidden[mask, :], val_targets[mask, task:task+1], kernel)
                gaussian.optimize()

                avg_test_preds, _ = gaussian.predict(
                    avg_last_hidden_test)

                predictions[:, task:task+1] = avg_test_preds
                confidence[:, task:task+1] = np.maximum(avg_test_preds, 1 - avg_test_preds)
        elif args.confidence == 'probability':
            predictions = avg_test_preds
            confidence = np.maximum(avg_test_preds, 1 - avg_test_preds)
        elif args.confidence == 'random_forest':
            predictions = np.ndarray(
                shape=(len(test_smiles), args.num_tasks))
            confidence = np.ndarray(
                shape=(len(test_smiles), args.num_tasks))

            val_targets = np.array(val_data.targets())

            n_trees = 100
            for task in range(args.num_tasks):
                forest = RandomForestClassifier(n_estimators=n_trees)

                mask = val_targets[:, task] != None
                forest.fit(avg_last_hidden[mask, :], val_targets[mask, task])

                avg_test_preds = forest.predict(avg_last_hidden_test)
                predictions[:, task] = avg_test_preds

                avg_test_var = fci.random_forest_error(
                    forest, avg_last_hidden[mask, :], avg_last_hidden_test) * (-1)
                confidence[:, task] = avg_test_var    
        elif args.confidence == 'conformal':
            predictions = avg_test_preds
            confidence = np.ndarray(
                shape=(len(test_smiles), args.num_tasks))

            val_targets = np.array(val_data.targets())
            for task in range(args.num_tasks):
                non_conformity = np.ndarray(shape=(len(val_targets)))

                for i in range(len(val_targets)):
                    non_conformity[i] = kNN(avg_last_hidden[i, :], val_targets[i, task], avg_last_hidden, val_targets[:, task])
                
                for i in range(len(test_smiles)):
                    alpha = kNN(avg_last_hidden_test[i, :], round(predictions[i, task]), avg_last_hidden, val_targets[:, task])

                    if alpha == None:
                        confidence[i, task] = 0
                        continue

                    non_null = non_conformity[non_conformity != None]
                    confidence[i, task] = np.sum(non_null >= alpha) / len(non_null)
        elif args.confidence == 'boost':        
            # Calculate Tanimoto Distances
            val_smiles = val_data.smiles()
            val_max_tanimotos = np.ndarray(shape=(len(val_smiles), 1))
            val_avg_tanimotos = np.ndarray(shape=(len(val_smiles), 1))
            val_new_substructs = np.ndarray(shape=(len(val_smiles), 1))
            test_max_tanimotos = np.ndarray(shape=(len(test_smiles), 1))
            test_avg_tanimotos = np.ndarray(shape=(len(test_smiles), 1))
            test_new_substructs = np.ndarray(shape=(len(test_smiles), 1))

            train_smiles_sfp = [morgan_fingerprint(s) for s in train_data.smiles()]
            train_smiles_union = [1 if 1 in [train_smiles_sfp[i][j] for i in range(len(train_smiles_sfp))] else 0 for j in range(len(train_smiles_sfp[0]))]
            for i in range(len(val_smiles)):
                temp_tanimotos = tanimoto(val_smiles[i], train_smiles_sfp, lambda x: x)
                val_max_tanimotos[i, 0] = max(temp_tanimotos)
                val_avg_tanimotos[i, 0] = sum(temp_tanimotos)/len(temp_tanimotos)

                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(val_smiles[i]))
                fp = morgan_fingerprint(smiles)
                val_new_substructs[i, 0] = sum([1 if fp[i] and not train_smiles_union[i] else 0 for i in range(len(fp))])
            for i in range(len(test_smiles)):
                temp_tanimotos = tanimoto(test_smiles[i], train_smiles_sfp, lambda x: x)
                test_max_tanimotos[i, 0] = max(temp_tanimotos)
                test_avg_tanimotos[i, 0] = sum(temp_tanimotos)/len(temp_tanimotos)

                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(test_smiles[i]))
                fp = morgan_fingerprint(smiles)
                test_new_substructs[i, 0] = sum([1 if fp[i] and not train_smiles_union[i] else 0 for i in range(len(fp))])
            
            model.use_last_hidden = True
            original_preds = predict(
                model=model,
                data=val_data,
                batch_size=args.batch_size,
                scaler=None
            )
            # Create and Train New Model
            features = (original_preds, val_max_tanimotos, val_avg_tanimotos, val_new_substructs)
            new_model = train_residual_model(np.concatenate(features, axis=1),
                                             original_preds,
                                             val_data.targets(),
                                             args.epochs)

            features = (avg_test_preds, test_max_tanimotos, test_avg_tanimotos, test_new_substructs)
            # confidence = new_model(np.concatenate(features, axis=1), 
                                    # avg_test_preds).detach().numpy()
            predictions = avg_test_preds
            confidence = np.abs(avg_test_preds - 0.5)
            # print(targets)
            # targets = np.extract(correctness > avg_correctness, targets).reshape((-1, args.num_tasks))
            # print(targets)
            # targets = (np.abs(avg_test_preds - targets) < 0.5) * 1
            

    if args.confidence:
        accuracy_log = {}
        if args.save_confidence:
            f = open(args.save_confidence, 'w+')

        for task in range(args.num_tasks):
            accuracy_sublog = []
            
            mask = targets[:, task] != None
            confidence_visualizations(args,
                                      predictions=np.extract(mask, predictions[:, task]),
                                      targets=np.extract(mask, targets[:, task]),
                                      confidence=np.extract(mask, confidence[:, task]),
                                      accuracy_sublog=accuracy_sublog)

            accuracy_log[args.task_names[task]] = accuracy_sublog

        if args.save_confidence:
            json.dump(accuracy_log, f)
            f.close()

    # Individual ensemble scores
    if args.show_individual_scores:
        for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            info(
                f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')

    return ensemble_scores

def tanimoto(smile, train_smiles_sfp, operation):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
    fp = morgan_fingerprint(smiles)
    morgan_sim = []
    for sfp in train_smiles_sfp:
        tsim = np.dot(fp, sfp) / (fp.sum() +
                                    sfp.sum() - np.dot(fp, sfp))
        morgan_sim.append(tsim)
    return operation(morgan_sim)

def confidence_visualizations(args: Namespace,
                              predictions: List[Union[float, int]] = [],
                              targets: List[Union[float, int]] = [],
                              confidence: List[Union[float, int]] = [],
                              accuracy_sublog = None):
    error = list(np.abs(predictions - targets))
    sets_by_confidence = sorted(
        list(zip(confidence, error, predictions, targets)), key=lambda pair: pair[0])
    sets_by_error = sorted(list(zip(confidence, error, predictions, targets)),
                           key=lambda pair: pair[1])

    metric_func = get_metric_func(metric=args.metric)

    if args.c_cutoff_discrete:
        print(
            f'----Cutoffs Generated Using "{args.confidence}" Estimation Process----')
        for i in range(10):
            c = int((i/10) * len(sets_by_confidence))
            kept = (len(sets_by_confidence) - c) / len(sets_by_confidence)

            accuracy = evaluate_predictions(
                preds=[[pair[2]] for pair in sets_by_confidence[c:]],
                targets=[[pair[3]] for pair in sets_by_confidence[c:]],
                num_tasks=1,
                metric_func=metric_func,
                dataset_type=args.dataset_type
            )

            log = f'Cutoff: {sets_by_confidence[c][0]:5.3f} {args.metric}: {accuracy[0]:5.3f} Results Kept: {int(kept*100)}%'
            print(log)

            if args.save_confidence:
                accuracy_sublog.append({'metric': args.metric,
                                         'accuracy': accuracy[0],
                                         'percent_kept': int(kept*100),
                                         'cutoff': sets_by_confidence[c][0]})

        print(f'----Cutoffs Generated by Ideal Confidence Estimator----')
        # Calculate Ideal Cutoffs
        for i in range(10):
            c = int((1 - i/10) * len(sets_by_error))
            kept = c / len(sets_by_error)

            accuracy = evaluate_predictions(
                preds=[[pair[2]] for pair in sets_by_error[:c]],
                targets=[[pair[3]] for pair in sets_by_error[:c]],
                num_tasks=1,
                metric_func=metric_func,
                dataset_type=args.dataset_type
            )

            print(f'{args.metric}: {accuracy[0]:5.3f}',
                  f'% Results Kept: {int(kept*100)}%')

    if args.c_cutoff:
        cutoff = []
        rmse = []
        square_error = [pair[1]*pair[1] for pair in sets_by_confidence]
        for i in range(len(sets_by_confidence)):
            cutoff.append(sets_by_confidence[i][0])
            rmse.append(np.sqrt(np.mean(square_error[i:])))

        plt.plot(cutoff, rmse)

    if args.c_bootstrap:
        # Perform Bootstrapping at 95% confidence.
        sum_subset = sum([val[0]
                          for val in sets_by_confidence[:args.bootstrap[0]]])

        x_confidence = []
        y_confidence = []

        for i in range(args.bootstrap[0], len(sets_by_confidence)):
            x_confidence.append(sum_subset/args.bootstrap[0])

            ys = [val[1] for val in sets_by_confidence[i-args.bootstrap[0]:i]]
            y_sum = 0
            for j in range(args.bootstrap[2]):
                y_sum += sorted(np.random.choice(ys,
                                                 args.bootstrap[1]))[-int(args.bootstrap[1]/20)]

            y_confidence.append(y_sum / args.bootstrap[2])
            sum_subset -= sets_by_confidence[i-args.bootstrap[0]][0]
            sum_subset += sets_by_confidence[i][0]

        plt.plot(x_confidence, y_confidence)

    if args.c_cutoff or args.c_bootstrap:
        plt.show()

    if args.c_histogram:
        scale = np.average(error) * 5

        for i in range(5):
            errors = []
            for pair in sets_by_confidence:
                if pair[0] < 2 * i or pair[0] > 2 * (i + 1):
                    continue
                errors.append(pair[1])
            plt.hist(errors, bins=10, range=(0, scale))
            plt.show()

    if args.c_joined_histogram:
        bins = 8
        scale = np.average(error) * bins / 2

        errors_by_confidence = [[] for _ in range(10)]
        for pair in sets_by_confidence:
            if pair[0] == 10:
                continue

            errors_by_confidence[int(pair[0])].append(pair[1])

        errors_by_bin = [[] for _ in range(bins)]
        for interval in errors_by_confidence:
            error_counts, _ = np.histogram(
                np.minimum(interval, scale), bins=bins)
            for i in range(bins):
                if len(interval) == 0:
                    errors_by_bin[i].append(0)
                else:
                    errors_by_bin[i].append(
                        error_counts[i] / len(interval) * 100)

        colors = ['green', 'yellow', 'orange', 'red',
                  'purple', 'blue', 'brown', 'black']
        sum_heights = [0 for _ in range(10)]
        for i in range(bins):
            label = f'Error of {i * scale / bins} to {(i + 1) * scale / bins}'

            if i == bins - 1:
                label = f'Error {i * scale / bins} +'
            plt.bar(list(
                range(10)), errors_by_bin[i], color=colors[i], bottom=sum_heights, label=label)
            sum_heights = [sum_heights[j] + errors_by_bin[i][j]
                           for j in range(10)]

        names = (
            f'{i} to {i+1} \n {len(errors_by_confidence[i])} points' for i in range(10))
        plt.xticks(list(range(10)), names)
        plt.xlabel("Confidence")
        plt.ylabel("Percent of Test Points")
        plt.legend()

        plt.show()

    if args.c_joined_boxplot:
        errors_by_confidence = [[] for _ in range(10)]
        for pair in sets_by_confidence:
            if pair[0] == 10:
                continue

            errors_by_confidence[int(pair[0])].append(pair[1])

        fig, ax = plt.subplots()
        ax.set_title('Joined Boxplot')
        ax.boxplot(errors_by_confidence)

        names = (
            f'{i} to {i+1} \n {len(errors_by_confidence[i])} points' for i in range(10))
        plt.xticks(list(range(1, 11)), names)
        plt.xlabel("Confidence")
        plt.ylabel("Absolute Value of Error")
        plt.legend()
        plt.show()

def kNN(x, y, values, targets):
    if y == None:
        return None
                  
    same_class_distances = []
    other_class_distances = []
    for i in range(len(values)):
        if np.all(x == values[i]) or targets[i] == None:
            continue
        
        distance = np.linalg.norm(x - values[i, :])
        if y == targets[i]:
            same_class_distances.append(distance)
        else:
            other_class_distances.append(distance)
    
    if len(other_class_distances) == 0 or len(same_class_distances) == 0:
        return None

    size = min([10, len(same_class_distances), len(other_class_distances)])
    return np.sum(heapq.nsmallest(size, same_class_distances)) / np.sum(heapq.nsmallest(size, other_class_distances))