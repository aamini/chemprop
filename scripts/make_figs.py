"""
make_figs_cleaned.py

"""

import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.integrate import simps
from sklearn import metrics
from tqdm import tqdm, trange
from pathos import multiprocessing # speeds up large map functions

from numpy import nan
from ast import literal_eval

METHOD_ORDER = ["evidence", "dropout", "ensemble", "sigmoid"]

METHOD_COLORS = {
    method: sns.color_palette()[index]
    for index, method in enumerate(METHOD_ORDER)
}

DATASET_MAPPING = {"lipo" : "Lipo",
                   "delaney" : "Delaney",
                   "freesolv" : "Freesolv",
                   "qm7": "QM7",
                   "bbbp" : "BBBP",
                   "sider" : "Sider",
                   "clintox" : "Clintox",
                   "hiv" : "HIV",
                   "tox21" : "Tox21",
                   "qm9" : "QM9",
                   "enamine" : "Enamine",
                   "qm9" : "QM9",
                   "enamine" : "Enamine",
                   "ppbr_az" : "PPBR",
                   "clearance_hepatocyte_az" : "Clearance",
                   "ld50_zhu" : "LD50",
                   }
# Datasets that we want to make ev tuning plots for
DATASETS = DATASET_MAPPING.values()

CLASSIF_SUMMARY_NAMES = ["avg_pr_entropy", "accuracy_entropy", "auc_roc_entropy",
                         "brier_entropy", "likelihood_entropy",
                         'Expected Probability', 'Predicted Probability']
REGR_SUMMARY_NAMES = ["rmse", "mae", "Predicted Probability",
                      "Expected Probability"]


def rename_method_df_none(df_column, rename_key):
    """ Transform method names """
    return [rename_key.get(i, None) for i in df_column]


def convert_dataset_names(dataset_series):
    """ Convert the dataset series in to the desired labeling"""
    ret_ar = []
    for i in dataset_series:
        ret_ar.append(DATASET_MAPPING.get(i, i))
    return ret_ar

def convert_to_std(full_df): 
    """ Convert confidence to std where applicable"""

    ## Convert all the confidence into std's
    new_confidence = full_df["stds"].values

    # At all the locations that *don't* have an std value, replace new
    # confidence with the value in confidence
    std_na = pd.isna(new_confidence)
    new_confidence[std_na] = full_df['confidence'][std_na].values
    full_df["confidence"] = new_confidence

def make_cutoff_table(df, outdir = "results/figures", out_name = "cutoff_table.txt",
                      export_stds=True, output_metric = "rmse",
                      table_data_name = "D-MPNN (RMSE)",
                      significant_best=True,
                      higher_better=False):
    """make_cutoff_table.

    Create the output latex results table and save in a text file.

    Args:
        df: Summary df of the data
        outdir: Save dir
        out_name: Name of outdirectory
        export_stds: If true, add \pm
        output_metric: Name of output metric to use. Default rmse.
        table_data_name: Str name of the data table. This can be used to
            distinguish this table as "Atomistic", "Classification", "High N",
            "Low N", etc. Defaults to "D-MPNN (RMSE)"
        significant_best: Bold best
        higher_better: If true, higher is better for this metric
    """

    # cutoffs = [0, 0.5, 0.75, 0.9, 0.95, 0.99][::-1]
    top_k = np.array([1, 0.5, 0.25, 0.1, 0.05])[::-1]

    df = df.copy()
    df["Method"] = df["method_name"]
    df["Data"] = convert_dataset_names(df["dataset"])

    uniq_methods = set(df["Method"].values)
    unique_datasets = set(df["Data"].values)

    data_order = [j for j in DATASETS if j in unique_datasets]
    method_order = [j for j in METHOD_ORDER if j in uniq_methods]

    table_items = []
    for data_group, data_df in df.groupby(["Data"]):
        for data_method, data_method_df in data_df.groupby(["Method"]):

            # Skip useless methods
            if data_method.lower() not in [method.lower() for method in method_order]:
                continue
            metric_sub = data_method_df[output_metric]
            # Total length of metric sub (use min for debugging)
            num_tested = np.min([len(i) for i in metric_sub])

            for cutoff in top_k:
                num_items = int(cutoff * num_tested)

                # NOTE: As above, the metric summary is already a mean over confidence
                # cutoffs and we should only take a point estimate from each trial run
                temp = np.reshape([j[-num_items] for j in metric_sub], -1)
                metric_cutoff_mean = np.mean(temp)
                metric_cutoff_std = stats.sem(temp) # standard error of the mean

                table_items.append({"Data" : data_group,
                                    "Method": data_method,
                                    "Cutoff" : cutoff,
                                    "METRIC_MEAN" : metric_cutoff_mean,
                                    "METRIC_STD": metric_cutoff_std})

    metric_summary = pd.DataFrame(table_items)
    means_tbl = pd.pivot_table(metric_summary,
                                values="METRIC_MEAN",
                                columns=["Cutoff"],
                                index=["Data", "Method"])

    stds_tbl = pd.pivot_table(metric_summary,
                                values="METRIC_STD",
                                columns=["Cutoff"],
                                index=["Data", "Method"])

    # Sort columns according to the cutoff values
    means_tbl = means_tbl.reindex(sorted(means_tbl.columns)[::-1], axis=1)
    stds_tbl = stds_tbl.reindex(sorted(stds_tbl.columns)[::-1], axis=1)

    output_tbl = means_tbl.astype(str)
    for cutoff in means_tbl.keys():
        cutoff_means = means_tbl[cutoff]
        cutoff_stds = stds_tbl[cutoff]

        for dataset in data_order:
            means = cutoff_means[dataset].round(5)
            stds = cutoff_stds[dataset]
            str_repr = means.astype(str)

            if export_stds:
                str_repr += " $\\pm$ "
                str_repr += stds.round(5).astype(str)

            if higher_better:
                if significant_best:
                    # significant_best finds the runs that are best by a
                    # statistically significant margin (ie. a standard dev)
                    METRIC_mins = means-stds
                    METRIC_maxs = means+stds
                    highest_metric_min = np.max(METRIC_mins)
                    best_methods = METRIC_maxs > highest_metric_min
                else:
                    # else, best is just the best mean performer
                    best_methods = (means == means.max())
            else:

                if significant_best:
                    # significant_best finds the runs that are best by a
                    # statistically significant margin (ie. a standard dev)
                    METRIC_mins = means-stds
                    METRIC_maxs = means+stds
                    smallest_metric_max = np.min(METRIC_maxs)
                    best_methods = METRIC_mins < smallest_metric_max
                else:
                    # else, best is just the best mean performer
                    best_methods = (means == means.min())

            # Bold the items that are best
            str_repr[best_methods] = "\\textbf{" + str_repr[best_methods] + "}"
            output_tbl[cutoff][dataset] = str_repr

    # Sort such that methods and datasets are in correct order
    output_tbl = output_tbl.reindex( pd.MultiIndex.from_product([ data_order,
                                                                 method_order]))

    assert(isinstance(table_data_name, str))
    output_tbl = output_tbl.set_index(pd.MultiIndex.from_product([[table_data_name],
                                                                  data_order,
                                                                  method_order]))
    # Write out
    with open(os.path.join(outdir, out_name), "w") as fp:
        fp.write(output_tbl.to_latex(escape=False))

def average_summary_df_tasks(df, avg_columns):
    """ Create averages of the summary df across tasks."""
    new_df = []

    # Columns to have after averaging
    keep_cols = ["dataset", "method_name", "trial_number"]
    subsetted = df.groupby(keep_cols)
    for subset_indices, subset_df in subsetted:
        return_dict = {}
        return_dict.update(dict(zip(keep_cols, subset_indices)))

        for column in avg_columns:
            task_values  = subset_df[column].values
            min_length = min([len(i) for i in task_values])

            new_task_values = []
            for j in task_values:
                j = np.array(j)
                if len(j) > min_length:
                    percentiles = np.linspace(0, len(j) - 1, min_length).astype(int)
                    new_task_values.append(j[percentiles])
                else:
                    new_task_values.append(j)
            avg_task = np.mean(np.array(new_task_values), axis=0).tolist()
            return_dict[column] = avg_task

        new_df.append(return_dict)

    return pd.DataFrame(new_df)

def evidence_tuning_plots(df, x_input = "Mean Predicted Avg",
                          y_input = "Empirical Probability",
                          x_name="Mean Predicted",
                          y_name="Empirical Probability"):
    """ Plot the tuning plot at different evidence values """

    def lineplot(x, y, trials, methods, **kwargs):
        """method_lineplot.

        Args:
            y:
            methods:
            kwargs:
        """
        uniq_methods = set(methods.values)
        method_order = sorted(uniq_methods)

        method_new_names = [f"$\lambda={i:0.4f}$" for i in method_order]
        method_df = []
        for method_idx, (method, method_new_name) in enumerate(zip(method_order,
                                                                   method_new_names)):
            lines_y = y[methods == method]
            lines_x = x[methods == method]
            for index, (xx, yy,trial) in enumerate(zip(lines_x, lines_y, trials)):

                to_append = [{x_name  : x,
                              y_name: y,
                              "Method": method_new_name,
                              "Trial" : trial}
                    for i, (x,y) in enumerate(zip(xx,yy))]
                method_df.extend(to_append)
        method_df = pd.DataFrame(method_df)
        x = np.linspace(0,1,100)
        plt.plot(x, x, linestyle='--', color="black")
        sns.lineplot(x=x_name, y=y_name, hue="Method",
                     alpha=0.8,
                     hue_order=method_new_names, data=method_df,)
        # estimator=None, units = "Trial")

    df = df.copy()
    # Query methods that have evidence_new_reg_2.0
    df = df[["evidence" in i for i in
             df['method_name']]].reset_index()

    # Get the regularizer and reset coeff
    coeff = [float(i.split("evidence_new_reg_")[1]) for i in df['method_name']]
    df["method_name"] = coeff
    df["Data"] = convert_dataset_names(df["dataset"])
    df["Method"] = df["method_name"]

    g = sns.FacetGrid(df, col="Data",  height=6, sharex = False, sharey = False)
    g.map(lineplot, x_input,  y_input, "trial_number",
          methods=df["Method"]).add_legend()

def plot_spearman_r(full_df, std=True): 
    """ Plot spearman R summary stats """

    if std: 
        convert_to_std(full_df)
    full_df["Data"] = convert_dataset_names(full_df["dataset"])

    grouped_df = full_df.groupby(["dataset", "method_name", "trial_number", "task_name"])
    spearman_r = grouped_df.apply(lambda x : stats.spearmanr(x['confidence'].values,  np.abs(x['error'].values )).correlation)

    new_df = spearman_r.reset_index().rename({0: "Spearman Rho" },
                                             axis=1)

    method_order = [i for i in METHOD_ORDER 
                    if i in pd.unique(new_df['method_name'])]
    new_df['Method'] = new_df['method_name']
    new_df['Dataset'] = new_df['dataset']

    plot_width = 2.6 * len(pd.unique(new_df['Dataset']))
    plt.figure(figsize=(plot_width, 5))

    sns.barplot(data=new_df , x="Dataset", y="Spearman Rho",
                hue="Method", hue_order = method_order)

    spearman_r_summary = new_df.groupby(["dataset", "method_name"]).describe()['Spearman Rho'].reset_index()
    return spearman_r_summary 

def make_tuning_plot_rmse(df, error_col_name="rmse",
                          error_title = "Top 10% RMSE",
                          cutoff = 0.10):

    """ Create the tuning plot for different lambda evidence parameters, but
    plot 10% RMSE instead of calibration. """

    df = df.copy()

    # Get the regularizer and reset coeff
    coeff = [float(i.split("evidence_new_reg_")[1]) if "evidence" in i else i for i in df['method_name']]
    df["method_name"] = coeff
    df["Data"] = convert_dataset_names(df["dataset"])
    df["Method"] = df["method_name"]

    # Get appropriate datasets
    trials = 'trial_number'
    methods = 'Method'

    # Make area plot
    uniq_methods = set(df["Method"].values)
    method_order = sorted(uniq_methods,
                          key=lambda x : x if isinstance(x, float) else -1)
    method_df = []
    datasets = set()
    for data, sub_df in df.groupby("Data"):
        # Add datasets
        datasets.add(data)
        rmse_sub = sub_df[error_col_name]
        methods_sub = sub_df["Method"]
        trials_sub= sub_df['trial_number']
        for method_idx, method in enumerate(method_order):
            # Now summarize these lines
            bool_select = (methods_sub == method)

            rmse_method = rmse_sub[bool_select]
            trials_temp = trials_sub[bool_select]
            areas = []
            # create area!
            for trial, rmse_trial in zip(trials_sub, rmse_method):
                num_tested = len(rmse_trial)
                cutoff_index = int(cutoff * num_tested) - 1
                rmse_val = rmse_trial[-cutoff_index]
                to_append = {error_title: rmse_val,
                             "Regularizer Coeff, $\lambda$": method,
                             "method_name": method,
                             "Data": data,
                             "Trial" : trial}
                method_df.append(to_append)
    method_df = pd.DataFrame(method_df)

    # Filter out dropout
    method_df = method_df[[i != "dropout" for i in
                           method_df['method_name']]].reset_index()

    # Normalize by dataset
    for dataset in datasets:
        # Make a divison vector of ones and change it to a different value only
        # for the correct dataset of interest to set max rmse to 1
        division_factor = np.ones(len(method_df))
        indices = (method_df["Data"] == dataset)

        # Normalize with respect to the ensemble so that this is 1
        max_val  = method_df[indices].query("method_name == 'ensemble'").mean()[error_title]

        # Take the maximum of the AVERAGE so it's normalized to 1
        division_factor[indices] = max_val
        method_df[error_title] = method_df[error_title] / division_factor

    method_df_evidence = method_df[[isinstance(i, float) for i in
                                    method_df['method_name']]].reset_index()
    method_df_ensemble = method_df[["ensemble" in str(i) for i in
                                    method_df['method_name']]].reset_index()

    data_colors = {
        dataset : sns.color_palette()[index]
        for index, dataset in enumerate(datasets)
    }

    min_x = np.min(method_df_evidence["Regularizer Coeff, $\lambda$"])
    max_x= np.max(method_df_evidence["Regularizer Coeff, $\lambda$"])

    sns.lineplot(x="Regularizer Coeff, $\lambda$", y=error_title,
                 hue="Data", alpha=0.8, data=method_df_evidence,
                 palette = data_colors)

    for data, subdf in method_df_ensemble.groupby("Data"):

        color = data_colors[data]
        area = subdf[error_title].mean()
        std = subdf[error_title].std()
        plt.hlines(area, min_x, max_x, linestyle="--", color=color, alpha=0.8)

    # Add ensemble baseline
    ensemble_line = plt.plot([], [], color='black', linestyle="--",
                                 label="Ensemble")
    # Now make ensemble plots
    plt.legend(bbox_to_anchor=(1.1, 1.05))

def make_area_plots(df, x_input = "Mean Predicted Avg",
                    y_input = "Empirical Probability"):
    """ Make evidence tuning plots """

    df = df.copy()

    # Get the regularizer and reset coeff
    coeff = [float(i.split("evidence_new_reg_")[1]) if "evidence" in i else i for i in df['method_name']]
    df["method_name"] = coeff
    df["Data"] = convert_dataset_names(df["dataset"])
    df["Method"] = df["method_name"]

    trials = 'trial_number'
    methods = 'Method'

    # Make area plot
    uniq_methods = set(df["Method"].values)
    method_order = sorted(uniq_methods,
                          key=lambda x : x if isinstance(x, float) else -1)
    method_df = []
    datasets = set()
    for data, sub_df in df.groupby("Data"):
        # Add datasets
        datasets.add(data)
        x_vals = sub_df[x_input]
        y_vals = sub_df[y_input]
        methods_sub = sub_df["Method"]
        trials_sub= sub_df['trial_number']
        for method_idx, method in enumerate(method_order):
            # Now summarize these lines
            bool_select = (methods_sub == method)
            lines_y = y_vals[bool_select]
            lines_x = x_vals[bool_select]
            trials_temp = trials_sub[bool_select]
            areas = []
            # create area!
            for trial, line_x, line_y in zip(trials_sub, lines_x, lines_y):
                new_y = np.abs(np.array(line_y) - np.array(line_x))
                area = simps(new_y, line_x)
                to_append = {"Area from parity": area,
                             "Regularizer Coeff, $\lambda$": method,
                             "method_name": method,
                             "Data": data,
                             "Trial" : trial}
                method_df.append(to_append)
    method_df = pd.DataFrame(method_df)
    method_df_evidence = method_df[[isinstance(i, float) for i in
                                    method_df['method_name']]].reset_index()
    method_df_ensemble = method_df[["ensemble" in str(i) for i in
                                    method_df['method_name']]].reset_index()
    data_colors = {
        dataset : sns.color_palette()[index]
        for index, dataset in enumerate(datasets)
    }

    min_x = np.min(method_df_evidence["Regularizer Coeff, $\lambda$"])
    max_x= np.max(method_df_evidence["Regularizer Coeff, $\lambda$"])

    sns.lineplot(x="Regularizer Coeff, $\lambda$", y="Area from parity",
                 hue="Data", alpha=0.8, data=method_df_evidence,
                 palette = data_colors)

    for data, subdf in method_df_ensemble.groupby("Data"):

        color = data_colors[data]
        area = subdf["Area from parity"].mean()
        std = subdf["Area from parity"].std()
        plt.hlines(area, min_x, max_x, linestyle="--", color=color, alpha=0.8)

    ensemble_line = plt.plot([], [], color='black', linestyle="--",
                                 label="Ensemble")
    # Now make ensemble plots
    plt.legend(bbox_to_anchor=(1.1, 1.05))

def save_plot(outdir, outname):
    """ Save current plot"""
    plt.savefig(os.path.join(outdir, "png", outname+".png"), bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "pdf", outname+".pdf"), bbox_inches="tight")
    plt.close()

def plot_calibration(df, x_input = "Mean Predicted Avg",
                     y_input = "Empirical Probability",
                     x_name="Mean Predicted",
                     y_name="Empirical Probability",
                     method_order = METHOD_ORDER, 
                     avg_x = False):
    """ plot_calibration.
    avg_x can be used to indicate that the x axis should be averaged position
    wise. That is, for classification calibration plots, we compute the
    confidence in different interval bands (e.g. we compute empirically the
    number of targets in the bin with predicted probability in the range of
    0.5,0.6). However, this average changes and the results are therefore very
    noisy. To enable averaging, we average across this. 

    """

    methods = df['method_name']
    uniq_methods = pd.unique(methods)
    method_order = [j for j in METHOD_ORDER if j in uniq_methods]
    method_df = []

    if avg_x: 
        df_copy = df.copy()
        new_list = [0]
        new_x_map = {}
        for method in uniq_methods: 
            temp_vals = df[df['method_name'] == method][x_input]
            new_ar = np.vstack(temp_vals)
            new_ar = np.nanmean(new_ar, 0) # avg columnwise
            new_x_map[method] = new_ar
        df_copy[x_input] = [new_x_map[method] for method in methods]
        df = df_copy

    x, y  = df[x_input].values, df[y_input].values


    method_df = [{x_name : xx, y_name : yy, "Method" : method}
                 for x_i, y_i, method in zip(x, y, methods)
                 for xx,yy in zip(x_i,y_i)]
    method_df = pd.DataFrame(method_df)
    sns.lineplot(x=x_name, y=y_name, hue="Method", alpha=0.8,
                 hue_order=method_order,
                 data=method_df,
                 palette = METHOD_COLORS)
    x = np.linspace(0,1,100)
    plt.plot(x, x, linestyle='--', color="black")

def conf_percentile_lineplot(df, error_col_name="rmse", error_title="RMSE",
                                   xlabel="Confidence Percentile",
                                   y_points = 1000,
                                   truncate_tail = False):
    """conf_percentile_lineplot.

    Args:
        df: Consolidated df of results
        error_col_name: Name of column in df that containse rror
        error_title: Renamed error in the output plot
        xlabel: xlabel
        y_points: Number of data points to plot
        truncate_tail: If true, don't plot the tail of the plot
    """

    methods = df['method_name']
    y = df[error_col_name]

    uniq_methods = set(methods.values)
    method_order = [j for j in METHOD_ORDER if j in uniq_methods]
    method_df = []

    # Truncate the last point if truncate_til is true
    slice_obj = slice(None) if not truncate_tail else slice(-1)

    lines = [line if len(line) < y_points
             else  np.array(line)[np.linspace(0, len(line) - 1, y_points).astype(int)]
             for line in y]

    method_df = [ {"Ratio": x, error_title : y, "Method" : method}
                 for line, method in zip(lines, methods)
                 for x, y in zip(np.linspace(0,1, len(line))[slice_obj] ,
                                 line[slice_obj])]
    method_df = pd.DataFrame(method_df)
    sns.lineplot(x="Ratio", y=error_title, hue="Method",
                 palette = METHOD_COLORS,
                 hue_order=method_order, data=method_df)

def distribute_task_plots(df, plot_fn):
    """ Distribute task plots"""


    num_tasks = len(pd.unique(df['task_name']))
    num_cols = int(min(num_tasks, 4))
    num_rows = int(np.ceil(num_tasks / num_cols))

    figsize = (20, 6*num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for task_index, (task_name, task_df) in enumerate(df.groupby("task_name")):
        row_num = task_index // (num_cols)
        col_num = task_index % num_cols
        if num_rows > 1:
            ax = axes[row_num, col_num]
        else:
            ax = axes[col_num]

        plt.sca(ax)
        plot_fn(task_df)
        ax.set_title(f"{task_name}")

        # Turn off x label
        if row_num != num_rows - 1:
            ax.set_xlabel("")

        # Turn off x label
        if col_num != 0:
            ax.set_ylabel("")

        # Remove legend unless upper right corner plot
        if col_num != num_cols -1 or row_num != 0:
            ax.get_legend().remove()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-data-file",
                        help="Loc of full data file")
    parser.add_argument("--summary-data-file",
                        help="""Loc of summary data file. Note: This is no longer
                        needed if it doesn't exist, make it""", default=None)
    parser.add_argument("--summary-data-task-file",
                        help="""Loc of summary data file for *tasks*.
                        Note: This is no longer needed if it doesn't exist,
                        make it""", default=None)
    parser.add_argument("--plot-type",
                        type=str,
                        choices= ["atomistic", "low_n", "classif", 
                                  "high_n", "tdc"],
                        default="atomistic")
    parser.add_argument("--outdir",
                        default="results/figures",
                        help="Directory to save figs")
    return parser.parse_args()

def make_atomistic_plots(full_df, summary_df, results_dir):
    """ make_atomistic_plots"""

    y_points = 150
    atomistic_rename = {"ensemble" : "ensemble",
                        "evidence_new_reg_0.2" : "evidence"}

    # Filter the dataset out for only the methods worth plotting
    # Rename in the process
    df_list = [full_df, summary_df]
    for df_index, df in enumerate(df_list):
        df['method_name'] = rename_method_df_none(df['method_name'],
                                                  rename_key = atomistic_rename)
        df_list[df_index] = df[[isinstance(i, str)
                                for i in df['method_name']]].reset_index(drop=True)

    # Make full df spearman r plot
    spearman_r_summary = plot_spearman_r(full_df, std=True)
    spearman_r_summary.to_csv(os.path.join(results_dir,
                                           f"spearman_r_atomistic_summary_stats.csv"))
    save_plot(results_dir, f"spearman_r_atomistic")

    plot_calibration(summary_df,
                     x_input="Expected Probability",
                     y_input = "Predicted Probability",
                     x_name="Expected Probability",
                     y_name = "Predicted Probability")
    save_plot(results_dir, "calibration_plot_atomistic")

    # Plot lineplots
    for error_name in ["mae", "rmse"]:
        conf_percentile_lineplot(summary_df, error_col_name=error_name,
                                 error_title=error_name.upper(),
                                 y_points=y_points, truncate_tail= True)
        save_plot(results_dir, f"atomistic_{error_name}")


        # Make latex cutoff tables
        make_cutoff_table(summary_df, outdir=results_dir,
                          table_data_name = f"Atomistic ({error_name.upper()})",
                          output_metric = error_name,
                          out_name = f"cutoff_table_atomistic_{error_name}.txt",
                          export_stds=True, significant_best=True,
                          higher_better= False)


def make_high_n_plots(full_df, summary_df, results_dir):
    """ make_high_n_plots"""

    y_points = 100
    high_n_rename = {"ensemble" : "ensemble",
                        "dropout": "dropout",
                        "evidence_new_reg_0.2" : "evidence"
                     }

    # Filter the dataset out for only the methods worth plotting
    # Rename in the process
    df_list = [full_df, summary_df]
    for df_index, df in enumerate(df_list):
        df['method_name'] = rename_method_df_none(df['method_name'],
                                                  rename_key = high_n_rename)
        df_list[df_index] = df[[isinstance(i, str)
                                for i in df['method_name']]].reset_index(drop=True)

    # Make full df spearman r plot
    spearman_r_summary = plot_spearman_r(full_df, std=True)
    spearman_r_summary.to_csv(os.path.join(results_dir,
                                           f"spearman_r_high_n_summary_stats.csv"))
    save_plot(results_dir, f"spearman_r_high_n")

    # Task averaged
    summary_df_task_avg =  average_summary_df_tasks(summary_df,
                                                    REGR_SUMMARY_NAMES)

    ### First, create plots that are _dataset_ specific
    ### Then create output tables
    for dataset_name, summary_df_sub in summary_df.groupby("dataset"):
        unique_tasks = pd.unique(summary_df_sub['task_name'])
        avg_summary_sub = summary_df_task_avg.query(f'dataset == "{dataset_name}"')

        # Create task specific break out plots
        if len(unique_tasks) > 1:

            ### Task specific calibration curves
            cal_plot_fn = lambda x : plot_calibration(x, x_input="Expected Probability",
                                                      y_input = "Predicted Probability",
                                                      x_name="Expected Probability",
                                                      y_name = "Predicted Probability")
            distribute_task_plots(summary_df_sub, cal_plot_fn)
            save_plot(results_dir, f"calibration_plot_high_n_{dataset_name}_task_plots")

            ### Task specific RMSE Curves
            rmse_plot_fn = lambda x : conf_percentile_lineplot(x, error_col_name="rmse", error_title="RMSE",
                                                                    y_points=y_points, truncate_tail= True)

            distribute_task_plots(summary_df_sub, rmse_plot_fn)
            save_plot(results_dir, f"rmse_plot_high_n_{dataset_name}_task_plots")

        plot_calibration(avg_summary_sub,
                         x_input="Expected Probability",
                         y_input = "Predicted Probability",
                         x_name="Expected Probability",
                         y_name = "Predicted Probability")
        save_plot(results_dir, f"calibration_plot_high_n_{dataset_name}")

        # Plot lineplots
        for error_name in ["rmse", "mae"]:
            conf_percentile_lineplot(avg_summary_sub, error_col_name=error_name,
                                     error_title=error_name.upper(),
                                     y_points=y_points, truncate_tail= True)
            save_plot(results_dir, f"high_n_{error_name}_{dataset_name}")

    # Plot lineplots
    for error_name in ["mae", "rmse"]:
        # Make latex cutoff tables
        make_cutoff_table(summary_df_task_avg, outdir=results_dir,
                          table_data_name = f"High N ({error_name.upper()})",
                          output_metric = error_name,
                          out_name = f"cutoff_table_high_n_{error_name}.txt",
                          export_stds=True, significant_best=True,
                          higher_better= False)

def make_low_n_plots(full_df, summary_df, results_dir):
    """ make_low_n_plots"""

    y_points = 100
    low_n_rename = {"ensemble" : "ensemble",
                        "dropout": "dropout",
                        "evidence_new_reg_0.2" : "evidence"
                     }

    # Before renaming make evidence tuning plots:
    # Verify that RMSE is steady across different evidential params
    make_tuning_plot_rmse(summary_df, error_col_name="rmse",
                          error_title = "Top 10% RMSE",
                          cutoff = 0.10)
    save_plot(results_dir, f"low_n_rmse_evidence_tuning")

    make_area_plots(summary_df, x_input="Expected Probability",
                    y_input = "Predicted Probability")
    save_plot(results_dir, f"low_n_evidence_tuning_plot_summary")

    # Make calibration plot for evidence
    evidence_tuning_plots(summary_df, x_input="Expected Probability",
                          y_input = "Predicted Probability",
                          x_name="Expected Probability",
                          y_name= "Predicted Probability")
    save_plot(results_dir, f"low_n_evidence_tuning_plot")

    ###
    # Filter the dataset out for only the methods worth plotting
    # Rename in the process
    df_list = [full_df, summary_df]
    for df_index, df in enumerate(df_list):
        df['method_name'] = rename_method_df_none(df['method_name'],
                                                  rename_key = low_n_rename)
        df_list[df_index] = df[[isinstance(i, str)
                                for i in df['method_name']]].reset_index(drop=True)
    full_df, summary_df = df_list

    spearman_r_summary = plot_spearman_r(full_df, std=True)
    spearman_r_summary.to_csv(os.path.join(results_dir,
                                           f"spearman_r_low_n_summary_stats.csv"))
    save_plot(results_dir, f"spearman_r_low_n")


    ### Now we should create each plot
    ### First, create plots that are _dataset_ specific
    ### Then create output tables..
    for dataset_name, summary_df_sub in summary_df.groupby("dataset"):
        ## First check if this is a task df
        unique_tasks = pd.unique(summary_df_sub['task_name'])

        plot_calibration(summary_df_sub,
                         x_input="Expected Probability",
                         y_input = "Predicted Probability",
                         x_name="Expected Probability",
                         y_name = "Predicted Probability")

        save_plot(results_dir, f"calibration_plot_low_n_{dataset_name}")

        # Plot lineplots
        conf_percentile_lineplot(summary_df_sub, error_col_name="rmse", error_title="RMSE",
                                       y_points=y_points, truncate_tail= True)
        save_plot(results_dir, f"low_n_rmse_{dataset_name}")

        conf_percentile_lineplot(summary_df_sub, error_col_name="mae", error_title="MAE",
                                       y_points=y_points, truncate_tail= True)
        save_plot(results_dir, f"low_n_mae_{dataset_name}")

    ### Plot _all_ evidence summary plots
    # Make latex cutoff tables
    make_cutoff_table(summary_df, outdir=results_dir,
                      table_data_name = "Low N (RMSE)",
                      output_metric = "rmse",
                      out_name = "cutoff_table_low_n_rmse.txt",
                      export_stds=True, significant_best=True,
                      higher_better= False)

    make_cutoff_table(summary_df, outdir=results_dir,
                      table_data_name = "Low N (MAE)",
                      output_metric = "mae",
                      out_name = "cutoff_table_low_n_mae.txt",
                      export_stds=True, significant_best=True,
                      higher_better = False)

def make_tdc_plots(full_df, summary_df, results_dir):
    """ make_low_n_plots"""

    y_points = 100
    low_n_rename = {"ensemble" : "ensemble",
                        "dropout": "dropout",
                        "evidence_new_reg_0.2" : "evidence"
                     }

    # Before renaming make evidence tuning plots:
    # Verify that RMSE is steady across different evidential params
    #make_tuning_plot_rmse(summary_df, error_col_name="rmse",
    #                      error_title = "Top 10% RMSE",
    #                      cutoff = 0.10)
    #save_plot(results_dir, f"low_n_rmse_evidence_tuning")

    #make_area_plots(summary_df, x_input="Expected Probability",
    #                y_input = "Predicted Probability")
    #save_plot(results_dir, f"low_n_evidence_tuning_plot_summary")

    # Make calibration plot for evidence
    #evidence_tuning_plots(summary_df, x_input="Expected Probability",
    #                      y_input = "Predicted Probability",
    #                      x_name="Expected Probability",
    #                      y_name= "Predicted Probability")
    #save_plot(results_dir, f"low_n_evidence_tuning_plot")

    ###
    # Filter the dataset out for only the methods worth plotting
    # Rename in the process
    df_list = [full_df, summary_df]
    for df_index, df in enumerate(df_list):
        df['method_name'] = rename_method_df_none(df['method_name'],
                                                  rename_key = low_n_rename)
        df_list[df_index] = df[[isinstance(i, str)
                                for i in df['method_name']]].reset_index(drop=True)
    full_df, summary_df = df_list

    spearman_r_summary = plot_spearman_r(full_df, std=True)
    spearman_r_summary.to_csv(os.path.join(results_dir,
                                           f"spearman_r_tdc__summary_stats.csv"))
    save_plot(results_dir, f"spearman_r_tdc")

    ### Now we should create each plot
    ### First, create plots that are _dataset_ specific
    ### Then create output tables..
    for dataset_name, summary_df_sub in summary_df.groupby("dataset"):
        ## First check if this is a task df
        unique_tasks = pd.unique(summary_df_sub['task_name'])

        # Plot calibration
        plot_calibration(summary_df_sub,
                         x_input="Expected Probability",
                         y_input = "Predicted Probability",
                         x_name="Expected Probability",
                         y_name = "Predicted Probability")

        save_plot(results_dir, f"calibration_plot_low_n_{dataset_name}")

        # Plot lineplots
        conf_percentile_lineplot(summary_df_sub, error_col_name="rmse", error_title="RMSE",
                                       y_points=y_points, truncate_tail= True)
        save_plot(results_dir, f"tdc_rmse_{dataset_name}")

        conf_percentile_lineplot(summary_df_sub, error_col_name="mae", error_title="MAE",
                                       y_points=y_points, truncate_tail= True)
        save_plot(results_dir, f"tdc_mae_{dataset_name}")

    ### Plot _all_ evidence summary plots
    # Make latex cutoff tables
    make_cutoff_table(summary_df, outdir=results_dir,
                      table_data_name = "Low N (RMSE)",
                      output_metric = "rmse",
                      out_name = "cutoff_table_tdc_x_rmse.txt",
                      export_stds=True, significant_best=True,
                      higher_better= False)

    make_cutoff_table(summary_df, outdir=results_dir,
                      table_data_name = "Low N (MAE)",
                      output_metric = "mae",
                      out_name = "cutoff_table_tdc_mae.txt",
                      export_stds=True, significant_best=True,
                      higher_better = False)

def make_classif_plots(full_df, summary_df, results_dir):
    """ make_classif_plots"""

    y_points = 100
    classif_rename= {"ensemble" : "ensemble",
                     "dropout": "dropout",
                     "evidence_new_reg_1.0" : "evidence",
                     "sigmoid" : "sigmoid"
                     }

    summary_df['method_name'] = rename_method_df_none(summary_df['method_name'],
                                                      rename_key = classif_rename)
    summary_df = summary_df[[isinstance(i, str)
                             for i in summary_df['method_name']]].reset_index(drop=True)

    ######
    name_pairs = [("accuracy_entropy", "Accuracy"),
                  ("avg_pr_entropy", "Avg Precision"),
                  ("brier_entropy", "Brier Score"),
                  ("likelihood_entropy", "Log Likelihood"),
                  ("auc_roc_entropy", "AUC ROC")]
    #name_pairs = [ ("avg_pr_entropy", "Avg Precision")]

    ## Split calibration_curves
    # Task averaged
    summary_df_task_avg =  average_summary_df_tasks(summary_df,
                                                    CLASSIF_SUMMARY_NAMES)

    ### Now we should create each plot
    ### First, create plots that are _dataset_ specific
    ### Then create output tables..
    for dataset_name, summary_df_sub in summary_df.groupby("dataset"):
        ## First check if this is a task df
        unique_tasks = pd.unique(summary_df_sub['task_name'])
        avg_summary_sub = summary_df_task_avg.query(f'dataset == "{dataset_name}"')

        # Create task specific break out plots
        if len(unique_tasks) > 1:

            ### Task specific calibration curves
            cal_plot_fn = lambda x : plot_calibration(x, x_input="Expected Probability",
                                                      y_input = "Predicted Probability",
                                                      x_name="Expected Probability",
                                                      y_name = "Predicted Probability",
                                                      avg_x = True)
            distribute_task_plots(summary_df_sub, cal_plot_fn)
            save_plot(results_dir, f"calibration_plot_classif_{dataset_name}_task_plots")

            ### Task specific avg_pr curves
            rmse_plot_fn = lambda x : conf_percentile_lineplot(x, error_col_name="avg_pr_entropy",
                                                                     error_title="Avg Precision",
                                                                     y_points=y_points, truncate_tail= True)
            distribute_task_plots(summary_df_sub, rmse_plot_fn)
            save_plot(results_dir, f"avg_precision_classif_{dataset_name}_task_plots")

        plot_calibration(avg_summary_sub, x_input="Expected Probability",
                         y_input = "Predicted Probability",
                         x_name="Expected Probability",
                         y_name = "Predicted Probability", 
                         avg_x = True)

        save_plot(results_dir, f"calibration_plot_classif_{dataset_name}")

        # Plot lineplots
        for col_name, error_title in name_pairs:
            conf_percentile_lineplot(avg_summary_sub,
                                          error_col_name=col_name,
                                          error_title=error_title,
                                          y_points=y_points, truncate_tail= True)
            save_plot(results_dir, f"classif_{dataset_name}_{col_name}")

    # Make latex cutoff tables
    make_cutoff_table(summary_df_task_avg, outdir=results_dir,
                      table_data_name = "Classif (Avg Pr)",
                      output_metric = "avg_pr_entropy",
                      out_name = "cutoff_table_classif_pr.txt",
                      export_stds=True, significant_best=True,
                      higher_better= True)

if __name__=="__main__":
    args = get_args()
    full_df = pd.read_csv(args.full_data_file, sep="\t", index_col=0)

    if args.plot_type == "classif":
        summary_df = pd.read_csv(args.summary_data_file, sep="\t", index_col=0)
        new_columns = {}
        for col in CLASSIF_SUMMARY_NAMES:
            new_col_vals = []
            for col_vals in summary_df[col]:
                eval_values = eval(col_vals)
                # eval_values = [i if not pd.isna(i) else 0 for i in eval_values]
                new_col_vals.append(eval_values)
            new_columns[col] = new_col_vals

        for col,val in new_columns.items():
            summary_df[col] = val
    else:
        summary_df = pd.read_csv(args.summary_data_file, sep="\t", index_col=0)
        new_columns = {}
        for col in REGR_SUMMARY_NAMES:
            new_col_vals = []
            for col_vals in summary_df[col]:
                if isinstance(col_vals, str):
                    if "," not in col_vals:
                        # Weird, seems like lists sometimes don't have commas
                        # when saved
                        eval_values = [float(i) for i in col_vals.strip()[1:-1].split()]
                    else:
                        eval_values = eval(col_vals)
                else:
                    eval_values = col_vals
                # eval_values = [i if not pd.isna(i) else 0 for i in eval_values]
                new_col_vals.append(eval_values)
            new_columns[col] = new_col_vals

        for col,val in new_columns.items():
            summary_df[col] = val

    results_dir = args.outdir
    os.makedirs(results_dir, exist_ok=True)

    sns.set(font_scale=1.3, style="white")

    results_dir  = os.path.join(results_dir, args.plot_type)

    for result_format in ["png", "pdf"]:
        os.makedirs(os.path.join(results_dir, result_format), exist_ok=True)


    full_df = full_df.query("partition == 'test'")

    if args.plot_type == "atomistic":
        make_atomistic_plots(full_df, summary_df, results_dir)
    elif args.plot_type == "high_n":
        make_high_n_plots(full_df, summary_df, results_dir)
    elif args.plot_type == "low_n":
        make_low_n_plots(full_df, summary_df, results_dir)
    elif args.plot_type == "tdc":
        make_tdc_plots(full_df, summary_df, results_dir)
    elif args.plot_type == "classif":
        make_classif_plots(full_df, summary_df, results_dir)
    else:
        raise NotImplementedError()
