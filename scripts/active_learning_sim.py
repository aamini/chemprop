""" Active learning simulation"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from scipy import stats
from scipy.stats import norm
from tqdm import tqdm
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.3, style="white", 
        rc={"figure.figsize" : (20,10)})

def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", 
                        default="data/enamine.csv",
                        help="Name of csv data file")
    parser.add_argument("--score-col-index", 
                        default=0,
                        type=int,
                        help="Score col index (add 1 to exclude smiles)")
    parser.add_argument("--seed", 
                        default=None,
                        type=int,
                        help="Seed")
    parser.add_argument("--save-folder", 
                        default="results/al_sim/",
                        type=str,
                        help="Name of out folder")

    ### Eval args
    parser.add_argument("--score-name", 
                        default="top_frac",
                        type=str,
                        choices=["top_frac"],
                        help="Name of score fn to use")
    parser.add_argument("--top-n", 
                        default=500,
                        type=int,
                        help="Fraction of top N scores to search")

    ### AL args
    parser.add_argument("--num_iters", 
                        default=10,
                        type=int,
                        help="Number of iterations")
    parser.add_argument("--init-frac", 
                        default=0.01,
                        type=float,
                        help="Fraction to initialize with")
    parser.add_argument("--total-select", 
                        default=0.1,
                        type=float,
                        help="Fraction of points to select in total")
    parser.add_argument("--strat", 
                        default="rand",
                        type=str,
                        choices= ["rand", "greedy", "lcb"], 
                        help="Type of strategy")
    parser.add_argument("--step-scale", 
                        default="log",
                        type=str,
                        choices= ["linear", "log"], 
                        help="Step scale for selection")

    ### Model Args
    parser.add_argument("--pred-type", 
                        default="exact",
                        type=str,
                        choices= ["exact", "norm"], 
                        help="Type of predictions")
    parser.add_argument("--pred-std-mean", 
                        default=0.1,
                        type=float,
                        help="Mean of std for prediction distribution")
    parser.add_argument("--pred-std-var", 
                        default=0.0,
                        type=float,
                        help=("Variance of std for pred distribution."
                              " Higher variance means steeper RMSE cutoff plots"))
    parser.add_argument("--conf-shift", 
                        default=1.0,
                        type=float,
                        help=("If < 1, model is overconfident. "
                              "If > 1, model is underconfident."))

    return parser.parse_args()

def load_data(data_file : str, 
              score_col_index: int) -> np.array: 
    """Return numpy arrays with """

    df = pd.read_csv(data_file)
    col_name = df.columns[score_col_index +1] 
    return df[col_name].values

def get_score_fn(score_name : str, data : np.array, top_n : int):
    """ get_score_fn. 

    Return a function of an array that scores the selected pool of scores. 

    """

    score_fn = None
    if score_name == "top_frac": 
        k_thresh_score = np.sort(data)[top_n]

        def score_fn(selected : np.array): 

            top_k_selected = np.sort(selected)[:top_n]
            percent_overlap = 100 * np.mean(top_k_selected <= k_thresh_score)
            return percent_overlap 
    else: 
        raise NotImplementedError()

    if score_fn is None: 
        raise NotImplementedError()
    return score_fn

def get_preds(pred_type : str, pool: np.array, 
              pred_std_mean : float, pred_std_var : float, 
              conf_shift : float, 
              LOWEST_CONF = 1e-9) -> (np.array, np.array): 
    """ Get predictions and confidence."""

    pool_shape_ones = np.ones(pool.shape) 
    if pred_type == "norm": 
        pred_std = np.random.normal(pool_shape_ones * pred_std_mean,
                                    pool_shape_ones * pred_std_var)

        # Make it a cutoff normal
        pred_std[pred_std <= 0] = LOWEST_CONF

        preds = np.random.normal(pool, pred_std)
        confs =  pred_std * conf_shift

        #confs[confs <= 1e-9] = LOWEST_CONF
        return (preds, confs)
    elif pred_type == "exact": 
        return (pool,  np.zeros(len(pool)))
    else: 
        raise NotImplementedError()

def pred_rmse(preds : np.array, trues : np.array): 
    """ Get rmse of predictions"""
    return np.sqrt(np.mean(np.square(trues - preds)))

def compute_calibration_curve(preds : np.array,
                              conf : np.array,
                              trues : np.array, 
                              num_partitions= 40):
    """ Compute calibration. """
    expected_p = np.arange(num_partitions+1)/num_partitions

    # Taken from github link in docstring
    norm = stats.norm(loc=0, scale=1)
    # At expected p of 0, we see 0.25 / 2.0 and 0.75 / 2.0
    gaussian_lower_bound = norm.ppf(0.5 - expected_p / 2.0)
    gaussian_upper_bound = norm.ppf(0.5 + expected_p / 2.0)

    residuals = preds - trues
    conf_ = conf.flatten() 
    #mask = (conf_ != 0 )
    #residuals = residuals[mask]
    #conf_ = conf_[mask]
    normalized_residuals = (residuals.flatten() / conf_).reshape(-1, 1)
    above_lower = normalized_residuals >= gaussian_lower_bound
    below_upper = normalized_residuals <= gaussian_upper_bound
    within_quantile = above_lower * below_upper
    obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)
    return expected_p, obs_proportions

def compute_rmse_curve(preds : np.array,
                       conf : np.array,
                       trues : np.array,
                       skip_factor = 500, 
                       ignore_last = True):
    """ Compute rmse plot. """
    indices = np.arange(len(preds))
    sorted_indices = sorted(indices, 
                            key= lambda x : conf[x], 
                            reverse=True)

    sorted_pred = preds[sorted_indices]
    sorted_conf = conf[sorted_indices]
    sorted_true = trues[sorted_indices]
    sorted_error = sorted_pred - sorted_true

    cutoff,errors = [], []
    error_list = [er**2 for er in sorted_error]

    total_error = np.sum(error_list)
    for i in tqdm(range(0, len(error_list), skip_factor)):
        cutoff.append(sorted_conf[i])

        if total_error < 0:
            #print(f"Total error is: {total_error}; setting to zero")
            total_error = 0

        errors.append(np.sqrt(total_error / len(error_list[i:])))
        total_error -= np.sum(error_list[i :i+skip_factor])


    if ignore_last: 
        errors = errors[:-1]

    conf_cutoff = np.linspace(0,1, len(errors))
    return conf_cutoff, np.array(errors)

def compute_model_props(data_file : str, score_col_index : int, 
                        seed : int = 0, pred_type : str = "norm", 
                        save_folder: str = "al_sim_out",
                        num_trials : int = 5, **kwargs): 

    """ Compute model props """
    loaded_data = load_data(data_file, score_col_index)
    if seed is not None: 
        np.random.seed(seed)

    pred_std_means =  [0.1, 0.2, 0.3, 0.5, 1.0] 
    pred_std_vars =  [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    conf_shifts =  [0.5, 0.75, 1, 1.25, 2]

    param_space = list(itertools.product(*[pred_std_means, 
                                           pred_std_vars,
                                           conf_shifts]))
    rmse_df = [] 
    cal_df = []
    for j in range(num_trials): 
        for pred_std_mean, pred_std_var, conf_shift in param_space: 
            model_name = (r"$\mu_{\sigma_p}$ = "
                          f"{pred_std_mean:02.2f},"
                          r" $\sigma_{\sigma_p}$ = "
                          f"{pred_std_var:02.2f}," 
                          rf" $c$ = {conf_shift:02.2f}")
            extra_args = {"pred_std_mean" : pred_std_mean, 
                          "pred_std_var" : pred_std_var, 
                          "conf_shift" : conf_shift}
            preds, conf = get_preds(pred_type = pred_type, pool = loaded_data, 
                                    pred_std_mean = pred_std_mean, 
                                    pred_std_var = pred_std_var,
                                    conf_shift = conf_shift)
            rmse_x, rmse_y = compute_rmse_curve(preds, conf, loaded_data)
            rmse_df.extend([{"Cutoff" : x, "RMSE": y, "Model" : model_name,
                             **extra_args} 
                            for x,y in zip(rmse_x, rmse_y)])
            calibration_x, calibration_y = compute_calibration_curve(preds, conf, loaded_data)
            cal_df.extend([{"Expected Prob" : x, "Observed Prob": y, 
                            "Model" : model_name, **extra_args } 
                           for x,y in zip(calibration_x, calibration_y)])

    cal_df = pd.DataFrame(cal_df)
    rmse_df = pd.DataFrame(rmse_df)
    cal_df.to_csv(os.path.join(save_folder, "calibration_df.csv"))
    rmse_df.to_csv(os.path.join(save_folder, "cutoff_df.csv"))


    ### Plot RMSE 
    #print("Making RMSE Plot")
    #plt.figure()
    #rmse_df = rmse_df.sort_values(by="Model")
    #sns.lineplot(data=rmse_df, x="Cutoff", y="RMSE", hue="Model")
    #plt.savefig(os.path.join(save_folder, "conf_cutoff.png"),
    #            bbox_inches="tight")
    #plt.close()

    #### Calibration Plot
    #print("Making Calibration Plot")

    #plt.figure()
    #cal_df = cal_df.sort_values(by="Model")
    #sns.lineplot(data=cal_df, x="Expected Prob", y="Observed Prob", hue="Model")
    #plt.plot(calibration_x, calibration_x, linestyle="--", color="black")
    #plt.savefig(os.path.join(save_folder, "calibration.png"),
    #            bbox_inches="tight")
    #plt.close()

def run_al(data_file : str, score_col_index : int, 
           init_frac: float, total_select : float, 
           seed : int = 0, num_iters : int = 10, 
           score_name : str = "top_frac", 
           top_n : int = 500, strat : str = "rand",
           pred_type : str = "norm", pred_std_mean : float= 0.1,
           pred_std_var : float = 0, step_scale : str = "log", 
           conf_shift : float = 0, **kwargs): 
    """ main. """

    loaded_data = load_data(data_file, score_col_index)

    if seed is not None: 
        np.random.seed(seed)

    score_fn = get_score_fn(score_name = score_name, 
                            data = loaded_data, top_n = top_n)

    ### Calculate num to select each iteration
    num_entries = len(loaded_data)
    init_frac = int(num_entries * init_frac)
    total_to_select = int(total_select * num_entries)

    # Copmute num to select
    if step_scale == "linear": 
        intervals = np.linspace(init_frac, total_to_select, 
                                num_iters+1).astype(int)
    elif step_scale == "log": 
        intervals = np.logspace(np.log10(init_frac), np.log10(total_to_select), 
                                num_iters+1).astype(int)
    else: 
        raise NotImplementedError()
    select_nums = np.diff(intervals)

    ###  Init
    random_perm = np.random.permutation(loaded_data)
    selected = random_perm[:init_frac]
    select_pool = random_perm[init_frac:]
    init_score = score_fn(selected)
    preds, conf = get_preds(pred_type = pred_type, pool = select_pool, 
                            pred_std_mean = pred_std_mean, 
                            pred_std_var = pred_std_var,
                            conf_shift = conf_shift)


    init_rmse = pred_rmse(preds, select_pool)

    scores = [init_score]
    num_selected = [init_frac]
    rmses = [init_rmse] 

    print(f"ITER 0-- SCORE : {scores[-1]:.2f}")
    print(f"ITER 0-- SELECTED : {num_selected[-1]} / {num_entries}")
    print(f"ITER 0-- MODEL ERROR : {rmses[-1]:.2f}\n")
    for index, iter_num in enumerate(range(1, num_iters + 1)): 

        num_to_select = select_nums[index]

        preds, conf = get_preds(pred_type = pred_type, pool = select_pool, 
                                pred_std_mean = pred_std_mean, 
                                pred_std_var = pred_std_var,
                                conf_shift = conf_shift)
        new_rmse = pred_rmse(preds, select_pool)

        new_selected = select_from_pool(strat=strat,
                                        num_to_select=num_to_select,
                                        pool=select_pool, 
                                        preds = preds,
                                        conf = conf)

        selected = np.hstack([selected, select_pool[new_selected]])
        select_pool = select_pool[~new_selected]
        scores.append(score_fn(selected))
        num_selected.append(len(selected))
        rmses.append(new_rmse)
        print(f"ITER {iter_num}-- SCORE : {scores[-1]:.2f}")
        print(f"ITER {iter_num}-- SELECTED : {num_selected[-1]} / {num_entries}")
        print(f"ITER {iter_num}-- MODEL ERROR : {rmses[-1]:.2f}\n")

    return np.array(num_selected) / num_entries, scores

def select_from_pool(strat: str, num_to_select : int, 
                     pool : np.array, preds : np.array = None, 
                     conf : np.array = None) -> np.array: 
    """ Select from a pool. 

        Return: 
            bool indices for selection 
    """

    if strat == "rand": 
        selected = np.zeros(len(pool))
        new_inds = np.random.choice(np.arange(len(pool)), 
                                    num_to_select, 
                                    replace=False)
        selected[new_inds] = 1
        selected = selected.astype(bool)

    elif strat == "greedy": 
        selected = np.zeros(len(pool))
        argsorted = sorted(np.arange(len(pool)), key = lambda x : preds[x])
        new_inds = np.array(argsorted)[:num_to_select]
        selected[new_inds] = 1
        selected = selected.astype(bool)

    elif strat == "lcb": 
        selected = np.zeros(len(pool))

        # Add confidence to get a lower bound (lower is better for selection)
        preds_modified = preds +  conf
        argsorted = sorted(np.arange(len(pool)), 
                           key = lambda x : preds_modified[x])
        new_inds = np.array(argsorted)[:num_to_select]
        selected[new_inds] = 1
        selected = selected.astype(bool)

    else: 
        raise NotImplementedError()

    return selected

if __name__=="__main__": 
    args = get_args()
    args = args.__dict__
    save_folder = args['save_folder']
    os.makedirs(save_folder, exist_ok=True)

    """ 
    Parameter search: 
    : Search Methods: 
    :   Greedy
    :   Random
    :   lcb
    : Model Params:  
    :   pred-std-mean (Avg RMSE)
    :   pred-std-var (How steep confidence is)
    :   pred-type (Norm or exact)
    :   conf-shift (How uncalibrated, >1 for under, < 1for over)

    The goal here is to simulate what a model predicts. Currently the model
    predictions are normal
    """
    compute_model_props(**args)

    if False:
        al_df = []

        pred_std_means = [0.1, 0.2, 0.3, 0.5, 1.0] 
        pred_std_vars = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
        conf_shifts = [0.5, 0.75, 1, 1.25, 2]
        strats = ["greedy", "lcb", "rand"] 
        param_space = list(itertools.product(*[pred_std_means, 
                                               pred_std_vars,
                                               conf_shifts, 
                                               strats]))
        result_df = []
        num_trials = 5
        for trial in tqdm(range(num_trials)):
            for pred_std_mean, pred_std_var, conf_shift, strat in tqdm(param_space): 
                extra_args = {"pred_std_mean" : pred_std_mean, 
                              "pred_std_var" : pred_std_var, 
                              "conf_shift" : conf_shift, 
                              "strat" : strat}
                args.update(extra_args)
                frac_selected, scores = run_al(**args)
                result_df.extend([{"FracSelected" : frac_select, 
                                   "Score" : score, **args} 
                                  for frac_select, score in zip(frac_selected, scores)])

        pd.DataFrame(result_df).to_csv(os.path.join(save_folder, "al_sims.csv"))


