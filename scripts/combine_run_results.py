import numpy as np
import argparse
import yaml
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=str, default=[], nargs="+",
                        help="regex to all result paths")
    args = parser.parse_args()

    #### 1. SPEARMAN ####
    all_spearman = {}
    for path in args.paths:
        with open(os.path.join(path, "fold_0", "spearman.txt"), "r") as p:
            # use safe_load instead load
            data = yaml.safe_load(p)

        all_spearman[path] = data

    rho_all = [data["rho"] for _, data in all_spearman.items()]
    p_all = [data["p"] for _, data in all_spearman.items()]
    print("\n".join([str((p, data["rho"])) for p, data in all_spearman.items()]))
    print("RHO: \t {} +/- {}".format(np.median(rho_all, 0), np.std(rho_all, 0)))
    print("P-VAL: \t {} +/- {}".format(np.median(p_all, 0), np.std(p_all, 0)))

    #### 2. CUTOFFS ####
    all_cutoffs = {}
    for path in args.paths:
        with open(os.path.join(path, "fold_0", "cutoff.txt"), "r") as p:
            # use safe_load instead load
            data = yaml.safe_load(p)

        all_cutoffs[path] = data
        style = "dashed" if len(all_cutoffs.keys()) > 9 else "solid"

        plt.plot(data["rmse"], label=path, linestyle=style)

    plt.plot(data["ideal_rmse"], label="ideal")
    plt.legend()
    plt.show()

    xx = np.linspace(0, 100, len(data["rmse"]))
    rmse_all = [data["rmse"] for _, data in all_cutoffs.items()]
    rmse_mean = np.mean(rmse_all, 0)
    rmse_std = np.std(rmse_all, 0)

    plt.fill_between(xx, rmse_mean-rmse_std, rmse_mean+rmse_std, alpha=0.5)
    plt.plot(xx, rmse_mean, label="mean")
    plt.plot(xx, data["ideal_rmse"], label="ideal")
    plt.legend()
    plt.show()
