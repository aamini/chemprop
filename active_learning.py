"""Train a model using active learning on a dataset."""
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import shutil
import time, datetime
import os, sys
import torch

from chemprop.parsing import parse_train_args
from chemprop.train.run_training import run_training, get_dataset_splits, get_atomistic_splits, evaluate_models
from chemprop.utils import create_logger


def ordered_list_diff(a, b):
    """ Returns the elements of a without any elements of b, while preserving
        order.

    :param a: list or array to remove elements from
    :param b: list or array to find and remove
    """
    list_diff = a[~np.in1d(a, b)]
    return list_diff


if __name__ == '__main__':
    args = parse_train_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)

    method = args.confidence
    dataset = os.path.basename(os.path.splitext(args.data_path)[0])
    results_root = Path(args.save_dir).parent #f"./al_results/{dataset}/{method}/"
    Path(results_root).mkdir(parents=True, exist_ok=True)


    for i_trial in range(args.num_folds):
        df = pd.DataFrame(
            columns=["Trial", "Train Data Ratio", "Score", "Uncertainty", "Entropy"])

        ### Load the data
        ## Atomistic network
        if args.atomistic:
            # Copy atomistic db to a temp folder
            if args.slurm_job and os.environ.get("TMPDIR") is not None:
                tmp_dir = os.environ.get("TMPDIR")
                _, file_name = os.path.split(args.data_path)
                old_loc = args.data_path
                new_loc = os.path.join(tmp_dir, file_name)
                shutil.copy2(args.data_path, new_loc)
                args.data_path = new_loc
                clean_up= lambda : os.remove(new_loc)

            (all_train_data, val_data, test_data), features_scaler, scaler = \
                get_atomistic_splits(args.data_path, args, logger)
        else:
            (all_train_data, val_data, test_data), features_scaler, scaler = \
                get_dataset_splits(args.data_path, args, logger)

        ### Define active learning step variables and subsample the tasks
        n_total = len(all_train_data)
        n_sample = n_total
        n_loops = args.num_al_loops

        ### Change active learning n_sample for early stopping
        if args.al_end_ratio is not None:
            if args.al_end_ratio > 1:
                raise ValueError("Arg al_end_ratio must be less than train size")
            total_data = len(all_train_data) + len(val_data) + len(test_data)
            early_stop_num = int(n_total * args.al_end_ratio)
            n_sample = early_stop_num

        n_start = int(n_total * args.al_init_ratio)

        if args.task_inds != []:
            all_train_data.sample_task_ind(args.task_inds)
            val_data.sample_task_ind(args.task_inds)
            test_data.sample_task_ind(args.task_inds)

            if scaler is not None:
                scaler.means = scaler.means[args.task_inds]
                scaler.stds = scaler.stds[args.task_inds]
        print(f"Ratio targets 0/1: {np.nanmean(np.array(all_train_data.targets(), dtype=np.float), axis=0)}")

        ### Compute the number of samples to use at each step of active learning
        if args.al_step_scale == "linear":
            n_samples_per_run = np.linspace(n_start, n_sample, n_loops)
        elif args.al_step_scale == "log":
            n_samples_per_run = np.logspace(np.log10(n_start), np.log10(n_sample), n_loops)
        else:
            raise ValueError(f"unknown args.al_step_scale = {args.al_step_scale}")
        n_samples_per_run = np.round(n_samples_per_run).astype(int)


        ### SLG: Move this to outside strategy loop to sample the same initial
        # batch per strategy
        train_subset_inds_start = np.random.choice(n_total, n_start, replace=False)
        for strategy in args.al_strategy:
            train_subset_inds = np.copy(train_subset_inds_start)

            tic_time = time.time() # grab the current time for logging

            ### Main active learning loop
            for i in range(n_loops):
                print(f"===> [{strategy}] Running trial {i_trial} with {n_samples_per_run[i]} samples")

                train_data = all_train_data.sample_inds(train_subset_inds)

                ### Train with the data subset, return the best models
                models = run_training(
                    train_data, val_data, scaler, features_scaler, args, logger)

                ### Sample according to a strategy
                if "explorative" in strategy or "score" in strategy or "exploit" in strategy:
                    ### Evaluate confidences on entire training set
                    all_train_data_unscaled = deepcopy(all_train_data)
                    if scaler is not None:
                        all_train_data_unscaled.set_targets(scaler.inverse_transform(all_train_data.targets()))

                    # Modified this line such that call with export_std flag, then grab the stds.
                    # will return: ensemble_scores, ensemble_predictions, confidence, std, entropy
                    all_train_scores, all_train_preds, all_train_conf, all_train_std, all_train_entropy = evaluate_models(
                        models, train_data, all_train_data_unscaled, scaler, args, logger, export_std=True)

                    ### Find the lowest confidence (highest unc) samples, add
                    # them to the training inds consider average entropy across
                    # tasks
                    sq_error = np.square(
                        np.array(all_train_data_unscaled.targets()) - all_train_preds)
                    rmse = np.sqrt( np.mean(sq_error.astype(np.float32), axis=1) )

                    # for evidence, either use std or conf for the uncertainty
                    if args.use_std:
                        mean_uncertainty = np.mean(all_train_std, axis=1)
                    else:
                        mean_uncertainty = np.mean(all_train_conf, axis=1)

                    if "explorative" in strategy:
                        per_sample_weight = mean_uncertainty
                    elif "score" in strategy:
                        per_sample_weight = rmse
                    elif "exploit" in strategy:
                        scaled_preds = scaler.transform(all_train_preds)
                        per_sample_weight = np.mean(scaled_preds, 1).astype(np.float32)

                        # Reverse and make sure weights (preds) are positive
                        if args.acquire_min:
                            per_sample_weight *= -1

                        std_mult = args.al_std_mult
                        if "_lcb" in strategy: # lower confidence bound
                            per_sample_weight += -std_mult * mean_uncertainty
                        elif "_ucb" in strategy: # upper confidence bound
                            per_sample_weight += +std_mult * mean_uncertainty
                        elif "_ts" in strategy: # thompson sampling
                            per_sample_weight = np.random.normal(
                                per_sample_weight, mean_uncertainty)

                        per_sample_weight -= per_sample_weight.min()

                    ### Save all the smiles along with their uncertainties/errors
                    train_subset_mask = np.zeros((n_total,))
                    train_subset_mask[train_subset_inds] = 1
                    df_scores = pd.DataFrame(data={
                        "Smiles": all_train_data.smiles(),
                        "Uncertainty": mean_uncertainty,
                        "Error": rmse,
                        "TrainInds": train_subset_mask
                    })
                    Path(os.path.join(results_root, "tracks")).mkdir(
                        parents=True, exist_ok=True)
                    df_scores.to_csv(os.path.join(results_root, "tracks",
                        f"{strategy}_step_{i}_{tic_time}.csv"))
                elif strategy == "random":
                    per_sample_weight = np.ones((n_total,)) # uniform
                else:
                    raise ValueError(f"Unknown active learning strategy {strategy}")

                ### Evaluate performance on test set and save
                evals_results = evaluate_models(models, train_data, test_data,
                                                scaler, args, logger,
                                                export_std=True, export_single_model_preds=True)
                if args.confidence:
                    test_scores, test_preds, test_conf, test_std, test_entropy, test_single_scores, test_single_preds = evals_results
                else:
                    test_scores, test_preds, test_conf, test_entropy, test_single_scores, test_single_preds = evals_results
                    test_std = test_conf = test_entropy = np.zeros_like(test_preds)


                ### Compute the top-k percent acquired
                # Grab the indicies that are in the top-k of only the training data
                top_k_scores_in_pool = np.sort(
                                    np.mean(all_train_data.targets(), 1))
                top_k_scores_in_pool = top_k_scores_in_pool[:args.al_topk] \
                                        if args.acquire_min else \
                                        top_k_scores_in_pool[-args.al_topk:]

                top_k_scores_in_selection = np.sort(
                                    np.mean(train_data.targets(), 1))
                top_k_scores_in_selection = top_k_scores_in_selection[:args.al_topk] \
                                        if args.acquire_min else \
                                        top_k_scores_in_selection[-args.al_topk:]

                # Find the overlap in indicies with our already acquired data points
                selection_overlap = np.in1d(top_k_scores_in_selection,
                                            top_k_scores_in_pool)

                # Compute the percent overlap
                percent_top_k_overlap = np.mean(selection_overlap) * 100
                ###


                ### Evaluate mae performmance
                args_other = deepcopy(args)
                args_other.metric = "mae" if args.metric == "rmse" else "rmse"
                if args.confidence:
                    test_scores_other, _, _, _, _, test_single_scores_other, test_single_preds_other = evaluate_models(
                        models, train_data, test_data, scaler, args_other, logger, export_std=True, export_single_model_preds=True)
                else:
                    test_scores_other, _, _, _, test_single_scores_other, test_single_preds_other = evaluate_models(
                        models, train_data, test_data, scaler, args_other, logger, export_std=True, export_single_model_preds=True)

                if args.confidence == "ensemble":
                    test_scores = test_single_scores
                    test_preds = test_single_preds
                    test_scores_other = test_single_scores_other
                    test_preds_other = test_single_preds_other


                df = df.append({
                    'Train Data Ratio': n_samples_per_run[i]/float(n_total),
                    'Score': np.mean(test_scores),
                    'Score_'+args_other.metric: np.mean(test_scores_other),
                    'Uncertainty': np.mean(test_conf),
                    'Standard Deviation': np.mean(test_std),
                    'Entropy': np.mean(test_entropy),
                    'Trial': i_trial,
                    'Strategy': strategy,
                    'Tasks': args.task_inds,
                }, ignore_index=True)


                ### Save the complete test performance (including uncs) to log
                test_error = test_preds - np.array(test_data.targets())
                log_data_dict = {
                    f"Error_{t}": test_error[:,t]
                    for t in range(test_error.shape[1])}

                log_data_dict.update({
                    "Smiles": test_data.smiles(),
                    "Uncertainty": np.mean(test_conf, 1),
                    "Entropy": np.mean(test_entropy, 1),
                    "Std": np.mean(test_std, 1),
                    "TopK": percent_top_k_overlap,
                    "Train Data Ratio": n_samples_per_run[i]/float(n_total),
                })
                df_test_log = pd.DataFrame(data=log_data_dict)
                Path(os.path.join(results_root, "scores")).mkdir(
                    parents=True, exist_ok=True)
                df_test_log.to_csv(os.path.join(results_root, "scores",
                    f"{strategy}_step_{i}_{tic_time}.csv"))

                logger.info("Percent top-k = {}".format(round(percent_top_k_overlap, 2)))

                ### Add new samples to training set
                n_add = n_samples_per_run[min(i+1, n_loops-1)] - n_samples_per_run[i]
                if n_add > 0: # n_add = 0 on the last iteration, when we are done

                    # Probability of sampling a new point, depends on the weight
                    per_sample_prob = deepcopy(per_sample_weight)

                    # Exclude data we've already trained with, and normalize to probability
                    per_sample_prob[train_subset_inds] = 0.0
                    per_sample_prob = per_sample_prob / per_sample_prob.sum()

                    # Sample accordingly and add to our training inds
                    if "sample" in strategy:
                        train_inds_to_add = np.random.choice(n_total, size=n_add, p=per_sample_prob, replace=False)
                    else:
                        # greedy, just pick the highest probability indicies
                        inds_sorted = np.argsort(per_sample_prob) # smallest to largest
                        train_inds_to_add = inds_sorted[-n_add:] # grab the last k inds

                    # Add the indices to the training set
                    train_subset_inds = np.append(train_subset_inds, train_inds_to_add)

                del models
                torch.cuda.empty_cache()

        # END SINGLE FOLD
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        df.to_csv(os.path.join(results_root, f"{timestamp}_{args.task_inds}.csv"))

    print(f"Done with all folds and saved into {results_root}")
    os._exit(1)
