from argparse import Namespace
import csv
from typing import List, Optional

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from .predict import predict
from .confidence_estimator import confidence_estimator_builder
from .confidence_evaluator import ConfidenceEvaluator
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, get_data_from_smiles, get_task_names
from chemprop.utils import load_args, load_checkpoint, load_scalers


def make_predictions(args: Namespace, smiles: List[str] = None) -> List[Optional[List[float]]]:
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: A list of lists of target predictions.
    """
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])
    orig_args = train_args

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Loading data')
    if smiles is not None:
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False)
    else:
        test_data = get_data(path=args.test_path, args=args, use_compound_names=args.compound_names, skip_invalid_smiles=False)

    if orig_args.confidence:
        print("Creating confidence esitmator using train args.")
        confidence_estimator = confidence_estimator_builder(orig_args.confidence)(MoleculeDataset([]),
                                                                             test_data, scaler,
                                                                             orig_args)
    else:
        print("Not creating confidence estimator")

    print('Validating SMILES')
    valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
    full_data = test_data
    test_data = MoleculeDataset([test_data[i] for i in valid_indices])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    test_smiles = test_data.smiles()

    if args.compound_names:
        compound_names = test_data.compound_names()
    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if train_args.features_scaling:
        test_data.normalize_features(features_scaler)

    # Predict with each model individually and sum predictions
    sum_preds = np.zeros((len(test_data), args.num_tasks))
    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        # Load model
        model = load_checkpoint(checkpoint_path, cuda=args.cuda)
        model_preds = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )

        train_args = load_args(args.checkpoint_paths[0])
        if orig_args.confidence:
            print("Note: Only using first argument for confidence constructor")
            confidence_estimator.process_model(model, predict)
        sum_preds += np.array(model_preds)

    # Ensemble predictions
    avg_preds = sum_preds / args.ensemble_size
    avg_preds = avg_preds.tolist()

    # Save predictions
    assert len(test_data) == len(avg_preds)
    print(f'Saving predictions to {args.preds_path}')

    # Put Nones for invalid smiles
    full_preds = [None] * len(full_data)
    for i, si in enumerate(valid_indices):
        full_preds[si] = avg_preds[i]
    avg_preds = full_preds
    test_smiles = full_data.smiles()


    # Compute entropy before we modify dropout for inference
    entropy = None
    if orig_args.confidence and orig_args.use_entropy:
        # Convert uncertainties from standard devs to entropy if desired
        print("Warning: Not outputting entropy in predictions")
        if args.dataset_type == 'classification':
            def categorical_entropy(p):
                return -(p*np.log(p) + (1-p)*np.log(1-p))
            entropy  = categorical_entropy(np.array(avg_preds))
        else:
            def gaussian_entropy(std):
                return -1/2.*np.log(2*np.pi*np.exp(1)*std**2)
            entropy = gaussian_entropy(np.array(avg_preds))

    # Export confidence and std
    if orig_args.confidence:
        (avg_preds, confidence) = confidence_estimator.compute_confidence(avg_preds)
        std = confidence_estimator.export_std()
        if std is None:
            std = confidence


    # Write predictions
    with open(args.preds_path, 'w') as f:
        writer = csv.writer(f)

        header = []
        header.append('smiles')
        if args.compound_names:
            header.append('compound_names')
        args.task_names = get_task_names(args.data_path)
        header.extend(args.task_names)

        # If the true targets are supplied in the test file.
        if test_data.targets() is not None:
            tasks_true = ["true_" + task for task in args.task_names]
            header.extend(tasks_true)

        if orig_args.confidence:
            # This is uncertainty, labeled as 'confidence' in repository
            header.append('uncertainty')
            header.append('std')

        writer.writerow(header)

        for i in range(len(avg_preds)):
            row = []
            smile = test_smiles[i]

            row.append(smile)
            if args.compound_names:
                row.append(compound_names[i])

            # Predicted values
            if avg_preds[i] is not None:
                row.extend(avg_preds[i])
            else:
                row.extend([''] * args.num_tasks)

            # True values
            if test_data.targets()[i] is not None:
                row.extend(test_data.targets()[i])

            if orig_args.confidence:
                # For consistency, use the same reverse meaning of uncertainty;
                # This actually means uncertainty (very likely predictions have low
                # uncertainty "confidence" values)
                row.extend(confidence[i])
                row.extend(std[i])

            writer.writerow(row)

    return avg_preds
