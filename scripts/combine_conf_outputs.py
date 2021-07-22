"""Combine all conf files from old predictions
"""

import os
import numpy as np
import argparse
#import yaml
import json
import pandas as pd
from tqdm import tqdm, trange
from pathos import multiprocessing # speeds up large map functions
from sklearn import metrics
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.integrate import simps

CONF_ORDERING = "sets_by_error"
SUMMARY_LOC = "fold_0"

def get_args():
    """ Get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", action="store",
                        help="""Directory containing different datasets of
                        results. E.g. dataset_dir/lipo/method/trial/conf.txt
                        and dataset_dir/logs_freesolv/method/trial#/conf.txt
                        should both be accessible""")
    parser.add_argument("--results-name", action="store",
                        default="conf.txt",
                        help="Name of files to actually merge")
    parser.add_argument("--summary-names", action="store",
                        default=[], nargs="+",
                        help="""List of all summary file names that should
                        appear. For example: --summary-names spearman.txt
                        log_likelihood.txt""")
    parser.add_argument("--outfile", action="store",
                        help="Prefix of all outfiles to be stored",
                        default="consolidated")
    parser.add_argument("--ood-test", action="store_true",
                        help="If true, collect ood test split info",
                        default=False)
    parser.add_argument("--result-type", action="store",
                        default="low_n",
                        help=("Type of experiments being combined. This will"
                              "change how to compute the summary df marginally"
                              "by setting the binning"),
                        choices= ["atomistic", "low_n", "classif", "high_n",
                                  "tdc"])

    args = parser.parse_args()
    return args


def extract_subdirs(path):
    """ Extract both the subdir names and full paths; exclude .files """

    # Avoid collecting .DS_Store and files (hack approach)
    dataset_names = [i for i in os.listdir(path)
                     if i[0] != "." and os.path.isdir(os.path.join(path, i))]
    dataset_dirs = [os.path.join(path, i) for i in dataset_names]

    return dataset_names, dataset_dirs

def extract_summary(summary_file, extra_info):
    """Get dict from summary file

    Args:
        summary_file: name of summary file
        extra_info: dict of extra info to add to each entry

    Return:
        new dict
    """

    # Switched to json load for speed
    temp_results = json.load(open(summary_file, "r"))
    temp_results.update(extra_info)
    return temp_results

def extract_yaml_info(conf_file, extra_info):
    """ Make one single list of dicts from the stored yaml.

    Args:
        conf_file: name of conf file
        extra_info: dict of extra info to add to each entry

    Return:
        list of new dicts
    """

    output_list = []
    # Switched to json load for speed
    temp_results = json.load(open(conf_file, "r"))

    # iterate through "test" and "val"
    for data_split in temp_results:

        # Now get each subdataset
        for task_name in temp_results[data_split]:
            # Note: Each dataset actually has dataset name in
            # it. Save this as well
            # Hard code selection of the ordering
            for entry in temp_results[data_split][task_name][CONF_ORDERING]:
                entry['partition'] = data_split
                entry['task_name'] = task_name
                # add extra info
                entry.update(extra_info)
                output_list.append(entry)

    return output_list

############

### Make summary of extracted confs 

def ordered_regr_error(df_subset, sort_factor, 
                       skip_factor=1, error_type = "mae"):
    """ Order df_subset by sort_factor and compute rmse or mae at each cutoff"""
    
    data = df_subset.to_dict('records')
    sorted_data = sorted(data,
                         key=lambda pair: pair[sort_factor],
                         reverse=True)
    cutoff,errors = [], []
    if error_type == "rmse":
        error_list = [set_['error']**2 for set_ in sorted_data]
    elif error_type == "mae":
        error_list = [np.abs(set_['error']) for set_ in sorted_data]
    else:
        raise NotImplementedError()

    total_error = np.sum(error_list)
    for i in tqdm(range(0, len(error_list), skip_factor)):
        cutoff.append(sorted_data[i][sort_factor])
        if error_type == "rmse": 
            errors.append(np.sqrt(total_error / len(error_list[i:])))
        elif error_type == "mae": 
            errors.append(total_error / len(error_list[i:]))
        else: 
            raise NotImplementedError()

        total_error -= np.sum(error_list[i :i+skip_factor])

    return np.array(errors)

def ordered_binary_class(df_subset, sort_factor, binary_fn =None,
                         skip_factor = 1):
    """ Order df_subset by sort_factor and compute binary_fn(targets, preds) at each cutoff. """
    data = df_subset.to_dict('records')
    sorted_data = sorted(data,
                         key=lambda pair: np.abs(pair[sort_factor]),
                         reverse=True)
    predictions = [i['prediction'] for i in sorted_data]
    targets = [i['target'] for i in sorted_data]

    #with multiprocessing.Pool(multiprocessing.cpu_count()//2) as p:
    #    results = p.map(
    #        lambda i: binary_fn(targets[i:], predictions[i:]),
    #        range(0, len(sorted_data), skip_factor))

    #print(f"Running binary class fn {binary_fn}")
    #print(f"Length of data: {len(df_subset)}")
    results = []
    for i in tqdm(range(0, len(sorted_data), skip_factor)): 
        results.append(binary_fn(targets[i:], predictions[i:]) )

    return results

def roc_auc_wrapper(targets, preds):
    """ AUC ROC wrapper"""
    if all([targets[0] == i for i in targets]):
        return 1
    else:
        return metrics.roc_auc_score(targets,preds)

def log_likelihood_wrapper(targets, preds, eps = 1e-15):
    """ Avg Log likelihood wrapper. """
    targs = np.array(targets)
    preds = np.array(preds)
    likelihood = np.sum(targs * np.log(preds + eps) + (1 - targs) * np.log(1 - preds + eps))
    return likelihood / len(targets)

def avg_pr_wrapper(targets, preds):
    """Avg pr wrapper"""
    if len(targets) <= 1 or all([0 == i for i in targets]):
        return 1
    else:
        return metrics.average_precision_score(targets,preds)

def brier_score(targets, preds):
    """Brier scorer"""
    targs = np.array(targets)
    preds = np.array(preds)
    return np.sum((targs - preds) **2 )/ len(targets)

def accuracy(targets, preds):
    """Accuracy"""
    targs = np.array(targets)
    preds = np.array(preds)
    preds[preds < 0.5]  = 0
    preds[preds >= 0.5] = 1
    return np.sum(preds == targs) / len(preds)

accuracy_entropy =  lambda x : ordered_binary_class(x, "entropy", accuracy)
accuracy_conf =  lambda x : ordered_binary_class(x, "confidence", accuracy)

cutoff_likelihood_entropy =  lambda x : ordered_binary_class(x, "entropy", log_likelihood_wrapper)
cutoff_likelihood_conf =  lambda x : ordered_binary_class(x, "confidence", log_likelihood_wrapper)

cutoff_roc_entropy =  lambda x : ordered_binary_class(x, "entropy", roc_auc_wrapper)
cutoff_roc_conf  =  lambda x : ordered_binary_class(x, "confidence", roc_auc_wrapper)

cutoff_brier_entropy =  lambda x : ordered_binary_class(x, "entropy", brier_score)
cutoff_brier_conf =  lambda x : ordered_binary_class(x, "confidence", brier_score)

cutoff_avg_pr_entropy =  lambda x : ordered_binary_class(x, "entropy", avg_pr_wrapper)
cutoff_avg_pr_conf =   lambda x : ordered_binary_class(x, "confidence", avg_pr_wrapper)



def make_summary_df(df, summary_functions, summary_names):
    """ Convert the full_df object into a summary df.

    Args:
        df: full df of all experiments
        summary_functions: fns to be applied to each experiment run (e.g. cutoff rmse)
        summary_names: Names of outputs in df for the summary functions
    """
    df = df.query("partition == 'test'")
    # Group by cutoff
    subsetted = df.groupby(["dataset", "method_name", "trial_number", 
                            "task_name"])
    merge_list = []
    for name, fn in tqdm(zip(summary_names, summary_functions), total=len(summary_names)):
        merge_list.append(subsetted.apply(fn).to_frame(name=name))
    summary_df = pd.concat(merge_list, axis=1).reset_index()

    return summary_df

def convert_to_std(full_df): 
    """ Convert confidence to std where applicable"""

    ## Convert all the confidence into std's
    new_confidence = full_df["stds"].values

    # At all the locations that *don't* have an std value, replace new
    # confidence with the value in confidence
    std_na = pd.isna(new_confidence)
    new_confidence[std_na] = full_df['confidence'][std_na].values
    full_df["confidence"] = new_confidence

### Calibration functions 

def classif_calibration_fn(df_subset, num_partitions = 10):
    """Compute calibration function for classification"""
    data = df_subset.to_dict('records')
    sorted_data = sorted(data,
                         key=lambda pair: np.abs(pair["prediction"]),
                         reverse=False)
    predictions = np.array([i['prediction'] for i in sorted_data])
    targets = np.array([i['target'] for i in sorted_data])

    boundaries = np.linspace(-0.000001,1,num_partitions + 1)

    indices = []
    for index , boundary in enumerate(boundaries[:-1]):
        indices.append(np.logical_and(predictions >  boundary,
                                      predictions <= boundaries[index +1]))
    pred_intervals = [predictions[index] for index in indices ]
    targ_intervals = [targets[index]  for index in indices ]

    # pred_intervals = np.array_split(predictions, num_partitions)
    # targ_intervals = np.array_split(targets, num_partitions)

    targ_probs = [np.mean(i) for i in targ_intervals]
    pred_probs = [np.mean(i) for i in pred_intervals]
    return (targ_probs, pred_probs)

def regr_calibration_fn(df_subset, num_partitions = 10):
    """ Create regression calibration curves in the observed bins.
    Full explanation and code taken from: 
    
    https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/89c42138d3028c8573a1a007ea8bef80ad2ed8e6/uncertainty_toolbox/metrics_calibration.py#L182

    """
    expected_p = np.arange(num_partitions+1)/num_partitions
    calibration_list = []
    method = df_subset["method_name"].values[0]

    df_subset = df_subset.query('partition == "test"')
    data = df_subset.to_dict('records')
    predictions = np.array([i['prediction'] for i in data])
    confidence = np.array([i['confidence'] for i in data])
    targets = np.array([i['target'] for i in data])
        
    # Taken from github link in docstring
    norm = stats.norm(loc=0, scale=1)
    gaussian_lower_bound = norm.ppf(0.5 - expected_p / 2.0)
    gaussian_upper_bound = norm.ppf(0.5 + expected_p / 2.0)
    residuals = predictions - targets
    normalized_residuals = (residuals.flatten() / confidence.flatten()).reshape(-1, 1)
    above_lower = normalized_residuals >= gaussian_lower_bound
    below_upper = normalized_residuals <= gaussian_upper_bound
    within_quantile = above_lower * below_upper
    obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)

    return obs_proportions

### 

def create_classification_summary(full_df, skip_factor = 30, 
                                  num_partitions = 6): 
    """ create_classification_summary."""
    summary_names = ["avg_pr_entropy", "accuracy_entropy", "auc_roc_entropy",
                     "brier_entropy", "likelihood_entropy"]
    summary_fns = [cutoff_avg_pr_entropy, accuracy_entropy, cutoff_roc_entropy,
                   cutoff_brier_entropy, cutoff_likelihood_entropy]
    summary_fns.insert(0, lambda x : classif_calibration_fn(x, num_partitions))
    summary_names.insert(0, "calibration_curves")

    summary_df = make_summary_df(full_df, summary_fns, summary_names)

    summary_df['Expected Probability'] = [i[0] for i in summary_df['calibration_curves']]
    summary_df['Predicted Probability'] = [i[1] for i in summary_df['calibration_curves']]
    summary_df.drop(labels="calibration_curves", axis=1, inplace=True)
    summary_names = summary_names[1:]
    summary_names.extend(['Expected Probability', 'Predicted Probability'])
    summary_df.fillna(0, inplace=True)
    return summary_df

def create_regression_summary(full_df, skip_factor = 1, num_partitions=40): 
    """ Given the regression full df as input, create a smaller summary by
    collecting data across trials

    Return:
        summary_df_names, summary_df

    """
    # Now make cutoff RMSE and MAE plots
    # Skip factor creates binning
    summary_names = ["rmse", "mae", "Predicted Probability", 
                     "Expected Probability"]
    cutoff_rmse = lambda x: ordered_regr_error(x, "confidence", 
                                               skip_factor = skip_factor,
                                               error_type="rmse")
    cutoff_mae = lambda x: ordered_regr_error(x, "confidence", 
                                              skip_factor = skip_factor, 
                                              error_type="mae")
    calibration_fn_ = lambda x : regr_calibration_fn(x, num_partitions = num_partitions)
    expected_p = lambda x : np.arange(num_partitions+1)/num_partitions
    summary_fns = [cutoff_rmse, cutoff_mae, calibration_fn_, expected_p]
    summary_df = make_summary_df(full_df, summary_fns, summary_names)
    # Delete this line 
    summary_df.fillna(0, inplace=True)
    return summary_names, summary_df

############

def main(saved_dir, data_file_name, outfile, summary_file_names, 
         result_type, ood_test = False): 
    """ main. """
    # Get all data
    full_data = []
    summary_data = []

    dataset_names, dataset_dirs = extract_subdirs(saved_dir)
    # Loop over different datasets
    for dataset_dir, dataset_name in zip(dataset_dirs, dataset_names):

        # Loop over all different methods in each dataset
        method_names, method_dirs = extract_subdirs(dataset_dir)

        # Loop over trials
        for method_dir, method_name in zip(method_dirs, method_names):

            # Loop over all trials in each directory
            trial_names, trial_dirs = extract_subdirs(method_dir)

            for trial_number, (trial_dir, trial_name) in tqdm(enumerate(zip(trial_dirs,
                                                                       trial_names))):

                extra_info = {"method_name" : method_name,
                              "trial_number" : trial_number,
                              "dataset" : dataset_name}

                # EXTRACT YAML
                results_file = os.path.join(trial_dir, data_file_name)
                if (os.path.isfile(results_file)):
                    new_data = extract_yaml_info(results_file, extra_info)
                    if ood_test:
                        ood_info = os.path.join(trial_dir, "fold_0/ood_info.csv")

                        if not os.path.isfile(ood_info): 
                            raise ValueError()

                        ood_df = pd.read_csv(ood_info, index_col=0)

                        smi = ood_df["smiles"].values
                        part = ood_df["partition"]
                        tani_sim = ood_df["max_sim_to_train"]

                        smi_to_part = dict(zip(smi, part))
                        smi_to_tani = dict(zip(smi, tani_sim))

                        for new_entry in new_data: 
                            part = new_entry['partition']
                            smi = new_entry['smiles']
                            ood_part = smi_to_part.get(smi, None)
                            tani = smi_to_tani.get(smi, None)
                            new_entry.update({'ood_partition' : ood_part, 
                                              'closest_sim' : tani})

                    full_data.extend(new_data)

                # EXTRACT SUMMARY FILES
                summary_dir = os.path.join(trial_dir, SUMMARY_LOC)
                current_summary = {}

                for summary_file_name in summary_file_names:
                    summary_file = os.path.join(summary_dir, summary_file_name)
                    if os.path.isfile(summary_file):
                        current_summary.update(extract_summary(summary_file, extra_info))

                summary_data.append(current_summary)

    df_full = pd.DataFrame(full_data)
    df_full.to_csv(f"{outfile}.tsv" , sep="\t")

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f"{outfile}_summary.tsv" , sep="\t")

    # Only select test set
    full_df = df_full.query("partition == 'test'").reset_index()

    ### Now compute summary df manually
    if result_type == "atomistic": 
        convert_to_std(full_df)
        summary_names, summary_df = create_regression_summary(full_df,
                                                              skip_factor = 30)
        summary_df.to_csv(f"{outfile}_summary_calc.tsv", sep="\t")
    elif result_type == "high_n": 
        convert_to_std(full_df)
        summary_names, summary_df = create_regression_summary(full_df,
                                                              skip_factor = 30)
        summary_df.to_csv(f"{outfile}_summary_calc.tsv", sep="\t")
    elif result_type == "low_n": 
        convert_to_std(full_df)
        summary_names, summary_df = create_regression_summary(full_df,
                                                              skip_factor = 1)
        summary_df.to_csv(f"{outfile}_summary_calc.tsv", sep="\t")
    elif result_type == "tdc": 
        convert_to_std(full_df)
        summary_names, summary_df = create_regression_summary(full_df,
                                                              skip_factor = 1)
        summary_df.to_csv(f"{outfile}_summary_calc.tsv", sep="\t")
    elif result_type == "classif": 
        summary_df = create_classification_summary(
            full_df, skip_factor = 30) 
        summary_df.to_csv(f"{outfile}_summary_calc.tsv", sep="\t")

if __name__=="__main__":
    args = get_args()
    saved_dir = args.dataset_dir
    data_file_name = args.results_name
    outfile = args.outfile
    summary_file_names = args.summary_names
    main(saved_dir, data_file_name, outfile, summary_file_names,
         args.result_type, args.ood_test)
