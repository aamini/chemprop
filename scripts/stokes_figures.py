import argparse
import os
import seaborn as sns
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

def plot_primary(args):
    """
    Plots the jointplot of model performance (predicted v. true) on held out
    test or validation sets from the primary Stokes Cell 2020 dataset.
    """
    for method in args.methods:

        files = glob.glob(os.path.join(args.preds_path,f"{method}/*.csv"))
        for fname in files:
            df = pd.read_csv(fname)
            smiles = df['smiles']
            preds = df['mean_inhibition']
            targets = df['true_mean_inhibition']

            # rmse = np.sqrt(np.mean((df['true_target'] - df['mean_inhibition'])**2))

            g = sns.jointplot(x='mean_inhibition', y='true_mean_inhibition', data=df, kind='reg')
            g.ax_joint.set_xlim(0,1.4)
            g.ax_joint.set_ylim(0,1.4)
            g.ax_joint.plot([0, 1.4], [0, 1.4], 'k--')

            save_name = fname.split('.csv')[0]
            save_path = save_name + str(args.use_stds) + '_performance' + args.ext
            plt.savefig(save_path)
            plt.show()
            plt.close()

def plot_broad(args):
    """
    Plots the jointplot of model performance (predicted v. true);
    jointplot of predictions v. uncertainty; as well as uncertainty cutoff v.
    experimentally validated hit rate, on Broad repurposing hub data.
    """

    for method in args.methods:
        files = glob.glob(os.path.join(args.preds_path,f"{method}/broad/*.csv"))
        df_percentiles = pd.DataFrame()
        df_percentiles_validated = pd.DataFrame()
        for i, fname in enumerate(files):
            save_name = fname.split('.csv')[0]
            df = pd.read_csv(fname)
            df_reduced = df.copy()

            ## HARDCODED
            smiles = df.iloc[:,0]
            activities = df.iloc[:,1]
            activity_label = df["mean_inhibition"]
            activity_true_label = df["true_mean_inhibition"]
            validated_activities = df.iloc[:,2]
            if args.use_stds:
                uncertainties = df.iloc[:,4]
                uncertainty_label = df.columns[4]
            else:
                uncertainties = df.iloc[:,3]
                uncertainty_label = df.columns[3]
            # stds = df.iloc[:,4]
            # std_label = df.columns[4]

            # HALICIN = "Nc1nnc(Sc2ncc(s2)[N+]([O-])=O)s1"
            # h_ind = smiles.loc[smiles == HALICIN].index.values[0]
            # plt.scatter(activities[h_ind], validated_activities[h_ind], s=100, c='r', zorder=1)
            # # print(np.log(uncertainties)[h_ind])
            #
            # inds_better = activities < activities[h_ind]
            # num_better = (inds_better).sum()
            # valid = ~np.isnan(validated_activities)
            # unc_better = uncertainties[valid*inds_better]
            # print("halacin percent confidence: ", (unc_better < uncertainties[h_ind]).mean())
            # import pdb; pdb.set_trace()

            #### PREDICTED V. TRUE ACTIVITIES
            plt.scatter(activities, validated_activities, c=np.log(uncertainties), s=60, zorder=2)
            plt.colorbar()
            ax = plt.gca()
            ax.set_xlabel('Predicted Mean Inhibition', fontsize = 12)
            ax.set_ylabel('True Mean Inhibition', fontsize = 12)

            save_path_performance = save_name + str(args.use_stds) + '_performance' + args.ext
            plt.savefig(save_path_performance)
            plt.show()
            plt.close()


            ##### JOINTPLOT UNCERTAINTY VS PREDICTED ACTIIVITY
            g = sns.jointplot(x=activity_label, y=uncertainty_label, data=df)
            ax = g.ax_joint
            ax.set_yscale('log')

            save_path_joint = save_name + str(args.use_stds) + '_joint_unc' + args.ext
            plt.savefig(save_path_joint)
            plt.close()

            #### CONSIDER ALL, PICK BEST PRED, FILTER UNCERTAINTY, EVALUATE VALIDATED OVERLAPS
            hit_rates = []
            ps = range(0, 51) # varying levels of uncertainty percentiles
            pred = np.array(activities)
            true = np.array(validated_activities)
            unc = np.array(uncertainties)

            arg_sort_pred = np.argsort(pred)

            k = args.k
            pred_topk = pred[arg_sort_pred[:k]]
            true_topk = true[arg_sort_pred[:k]]
            unc_topk = unc[arg_sort_pred[:k]]
            # import pdb; pdb.set_trace()
            def remove_nans(x):
                return x[~np.isnan(x)]

            hit_rate_no_unc = np.mean(remove_nans(true_topk) < 0.2) #0.2 cutoff from paper

            for p in ps:
                unc_thresh = np.percentile(unc_topk, p)
                true_topk_with_unc = true_topk[unc_topk <= unc_thresh]
                hit_rate_with_unc = np.mean(remove_nans(true_topk_with_unc) < 0.2) #0.2 cutoff from paper

                hit_rates.append(
                    (hit_rate_no_unc, hit_rate_with_unc))

                if len(remove_nans(true_topk[unc_topk <= unc_thresh])) >= 2:
                    df_percentiles = df_percentiles.append({
                        'Percentile': 100-p,
                        'Hit Rate': hit_rate_with_unc,
                        'Method': 'uncertainty',
                        'Trial': i,
                    }, ignore_index=True)
                df_percentiles = df_percentiles.append({
                    'Percentile': 100-p,
                    'Hit Rate': hit_rate_no_unc,
                    'Method': 'no uncertainty',
                    'Trial': i,
                }, ignore_index=True)

            sns.lineplot(x='Percentile', y='Hit Rate', style='Method', data=df_percentiles[df_percentiles["Trial"]==i])
            save_path_percentiles = save_name + str(args.use_stds) + '_percentile_0.2_trial_' + str(i) + args.ext
            plt.savefig(save_path_percentiles)
            plt.close()

        sns.lineplot(x='Percentile', y='Hit Rate', style='Method', hue='Trial', data=df_percentiles)
        save_path_hits= save_name + str(args.use_stds) + '_hit_rate_0.2_trials' + args.ext
        plt.savefig(save_path_hits)
        # plt.show()
        plt.close()

        # fig, ax = plt.subplots(figsize=(6, 6))
        sns.lineplot(x='Percentile', y='Hit Rate', hue='Method', data=df_percentiles)
        save_path_hits= save_name + str(args.use_stds) + '_hit_rate_0.2' + args.ext
        plt.savefig(save_path_hits, bbox_inches="tight")
        plt.show()
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the Stokes primary data results.')
    parser.add_argument('--methods', type=str, nargs='*', default=['evidence'],
                        help='methods for which to plot results')
    parser.add_argument('--preds_path', type=str, default='./stokes_results',
                        help='path to directory of results')
    parser.add_argument('--k', type=int, default=50,
                        help='number of top predictions (rank order) to consider')
    parser.add_argument('--use_stds', action='store_true', default=False,
                        help='Use standard deviations')
    parser.add_argument('--ext', type=str, default='.pdf',
                        help='type of plot to save')
    args = parser.parse_args()
    # plot_primary(args)
    plot_broad(args)
