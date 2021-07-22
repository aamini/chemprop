import argparse
import itertools
import seaborn as sns
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Plot the active learning results')
parser.add_argument('--path',
                    type=str,
                    nargs="+",
                    default="./al_results",
                    help='the path to the active learning results')
parser.add_argument('--cutoff',
                    type=float,
                    default="0.0",
                    help='cutoff ratio for most confident samples')
parser.add_argument('--topk',
                    action="store_true",
                    help='to use topk as metric')
parser.add_argument('--outname', action="store", default=None, help='outname')

args = parser.parse_args()

df = pd.DataFrame()

for run_root in args.path:
    dataset_paths = sorted(glob.glob(os.path.join(run_root, "*/")))
    datasets = [
        os.path.basename(os.path.dirname(path)) for path in dataset_paths
    ]

    for dataset_path, dataset in zip(dataset_paths, datasets):
        print(dataset)

        method_paths = glob.glob(os.path.join(dataset_path, "*/"))
        methods = [
            os.path.basename(os.path.dirname(path)) for path in method_paths
        ]

        for method_path, method in zip(method_paths, methods):
            files = glob.glob(os.path.join(method_path, "scores", "*.csv"))
            print(method, len(files))

            for fname in tqdm(files):
                df_single = pd.read_csv(fname)

                # Extract some data from the filename for plotting
                strategy = os.path.basename(fname).split("_step_")[0]
                trial_time = os.path.basename(fname).split("_")[-1].split(
                    ".csv")[0]
                train_ratio = np.round(df_single["Train Data Ratio"].iloc[0],
                                       7)

                # if strategy not in ["random", "exploit", "exploit_lcb"]:
                #     continue

                new_dict = {
                    'Train Data Ratio': train_ratio,  #train_ratio,  #50000*0.9*train_ratio,
                    'Trial Time': trial_time,
                    'Strategy': strategy,
                    'Method': method,
                    'Dataset': dataset
                }

                if args.topk:
                    new_dict.update({'top-k': df_single["TopK"].iloc[0]})
                else:
                    uncertainties = df_single["Std"]
                    threshold = uncertainties.quantile(args.cutoff)

                    df_cutoff = df_single[uncertainties >= threshold]
                    pred_error_keys = [
                        k for k in df_cutoff.keys() if "Error_" in k
                    ]
                    pred_errors = df_cutoff[pred_error_keys]
                    rmse = np.sqrt(np.mean(pred_errors**2, axis=0))
                    mae = np.mean(np.abs(pred_errors), axis=0)

                    new_dict.update({
                        'RMSE': rmse.mean(),
                        'MAE': mae.mean(),
                    })

                df = df.append(new_dict, ignore_index=True)

method_order = ["evidence_new_reg_0.2", "dropout", "ensemble", "nn"]
if not args.topk:
    df_improvement_ = pd.DataFrame()
    for method in df["Method"].unique():
        df_method = df[df["Method"] == method]
        df_method_rand = df_method[df_method["Strategy"] == "random"]
        df_method_exp = df_method[df_method["Strategy"] ==
                                  "explorative_greedy"]

        avg_random = df_method_rand.groupby("Train Data Ratio").mean()
        avg_random.reset_index(level=0, inplace=True)

        for index, row in df_method_exp.iterrows():
            x = row["Train Data Ratio"]
            match = avg_random[avg_random["Train Data Ratio"] == x]
            if len(match) == 0:
                continue
            rand = float(match["RMSE"])
            exp = row["RMSE"]
            pdec = (rand - exp) / rand * 100
            row["Percent Improvement"] = pdec

            df_improvement_ = df_improvement_.append(row, ignore_index=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df_improvement_,
                 x="Train Data Ratio",
                 y="Percent Improvement",
                 hue="Method",
                 marker="o",
                 hue_order=method_order)
    ax.grid()
    if args.outname is not None:
        plt.savefig(args.outname + "_improvement" + ".pdf",
                    bbox_inches="tight")
    plt.show()

metric = "top-k" if args.topk else "RMSE"
fig, ax = plt.subplots(figsize=(6, 4))
# g = sns.FacetGrid(df, col="Dataset", height=6, sharex=False, sharey=False)
# g.map_dataframe(sns.lineplot, "Train Data Ratio", metric, hue="Method", style="Strategy", marker="o").add_legend()
g = sns.lineplot(x="Train Data Ratio",
                 y=metric,
                 hue="Method",
                 style="Strategy",
                 marker="o",
                 hue_order=method_order,
                 data=df,
                 ax=ax)
ax.grid()

#g = sns.FacetGrid(df, col="Method", height=6, sharex=False, sharey=False)
#g.map_dataframe(sns.lineplot, "Train Data Ratio", metric, style="Strategy", marker="o").add_legend()

# [ax.grid() for a in g.axes for ax in a]

if args.outname is not None:
    plt.savefig(args.outname + "_score" + ".pdf", bbox_inches="tight")
plt.show()

from scipy.stats import ttest_ind
df_pair_ps = pd.DataFrame()
df_pair_fc = pd.DataFrame()
for x in df["Train Data Ratio"].unique():
    df_x = df[df["Train Data Ratio"] == x]
    df_x = df_x[df_x["Strategy"] == "exploit_lcb"]

    methods = df_x["Method"].unique()
    method_pairs = itertools.combinations(methods, 2)
    for method_a, method_b in method_pairs:
        a = df_x[df_x["Method"] == method_a]["top-k"]
        b = df_x[df_x["Method"] == method_b]["top-k"]
        _, pval = ttest_ind(a, b)
        df_pair_ps = df_pair_ps.append(
            {
                "Train Data Ratio": x,
                "Pair": f"{method_a}_{method_b}",
                "p-value": pval
            },
            ignore_index=True)

        fcs = np.array(a).reshape(-1, 1) / np.array(b).reshape(1, -1)
        fcs = fcs.flatten()
        df_pair_fc = df_pair_fc.append([{
            "Train Data Ratio": x,
            "Pair": f"{method_a}_{method_b}",
            "Fold Change": fc
        } for fc in fcs],
                                       ignore_index=True)

fc_means = df_pair_fc.groupby(["Train Data Ratio", "Pair"]).mean()
fc_stds = df_pair_fc.groupby(["Train Data Ratio", "Pair"]).std()
ps = df_pair_ps.groupby(["Train Data Ratio", "Pair"]).mean()
print("Fold Change Means:")
print(fc_means)
print("\nFold Change STDs:")
print(fc_stds)
print("\np-values:")
print(ps)

import pdb
pdb.set_trace()

# ### Plot the results
# sns.lineplot(
#     x="Train Data Ratio", y="RMSE", hue="Method",
#     style="Strategy", hue_order=methods, marker="o", data=df)
# plt.title(f"RMSE {dataset}")
# # plt.savefig(os.path.join(run_root, dataset, f"scores_overlay_{task}.pdf"))
# plt.show()
# plt.close()
#
# # sns.lineplot(
# #     x="Train Data Ratio", y="MAE", hue="Method",
# #     style="Strategy", hue_order=methods, marker='o', data=df)
# # plt.title(f"MAE {dataset}")
# # # plt.savefig(os.path.join(args.path, dataset, f"scores_overlay_{task}.pdf"))
# # plt.show()
# # plt.close()
