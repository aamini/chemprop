""" Pythonic launcher.

This now runs with a config file and copies the config file into the results
directory. If it already exists, copy it over with a new number extension. 

Sample run: 
    python scripts/run_jobs_slurm.py configs/reproducibility/low_n_demo.json

"""

import os
import shutil
import argparse
import subprocess 
import typing
from typing import List, Optional, Tuple
import time 
from datetime import datetime
import json

def get_args() -> Tuple[dict, str]: 
    """ get_args.

    Return: 
        Tuple[dict,str]: Args and name of file used 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Name of configuration file")
    args = parser.parse_args()
    print(f"Loading experiment from: {args.config_file}\n")
    args_new = json.load(open(args.config_file, "r"))
    return args_new, args.config_file

def dump_config_file(save_dir : str, config : str): 
    """ dump_config_file.

    Try to dump the output config file continuously. If it doesn't work,
    increment it.  

    Args:
        save_dir (str): Name of the save dir where to put this
        config (str): Location of the config file
    """

    # Dump experiment
    new_file = "experiment.config"
    config_path = os.path.join(save_dir, new_file)
    ctr = 1
    os.makedirs(save_dir, exist_ok=True)
    # Keep incrementing the counter
    while os.path.exists(config_path): 
        new_file =  f"experiment_{ctr}.config"
        config_path = os.path.join(save_dir, new_file)
        ctr += 1

    shutil.copy2(config, config_path)
    TIME_STAMP_DATE = datetime.now().strftime("%m_%d_%y")

    # Open timestamps file
    with open(os.path.join(save_dir, "timestamps.txt"), "a") as fp: 
          fp.write(f"Experiment {new_file} run on {TIME_STAMP_DATE}.\n")

def main(seeds : List[int] = range(10), n_ensembles : int = 5, 
         datasets : List[str] = ["lipo"], dataset_types = ["regression"] , 
         dataset_splits : List[str] = ["random"], 
         methods : List[str] = ["evidence_new_reg"], 
         reg_coefs : List[float] = [0.2],   
         split_sizes: List[float] = [0.8,0.1,0.1],
         save_dir : str = "results/submission_results_paper", 
         dropout : float = 0.1, 
         epochs: float = 100, 
         ensemble_threads: int = 2, 
         debug : bool = False, 
         use_gpu: bool = True, 
         job_name_prefix : str = "gnn_job", 
         experiment_name : str = "chemprop", 
         num_minutes : int = 30, 
         num_hours : int= 1, 
         num_days : int= 0, 
         use_slurm : bool = True, 
         atomistic : bool = False, 
         experiment_file_name: Optional[str]= None,
         max_lr_atomistic : float = 0.0002, 
         no_smiles_export : bool = False,
         final_lr_atomistic : float = 1e-4): 
    """ main.

    Run the launching procedure to spin up different Chemprop trials.

    Args: 
        seeds (List[int]): List of the seeds to run this fn with. Default
            range(10)
        n_ensembles (int): Number of ensembles to use when running UQ methods
            that take a set number of ensembles (e.g., dropout and ensemble).
            Default 5. 
        datasets (List[str]): List of datasets to run! Default ['lipo'] 
        dataset_types (List[str]): List of dataset types to use. 
            Default ['regression']. Options are "regression" and
            "classification". Must be same length as datasets
        dataset_splits (List[str]): List of dataset splits. Must be same size
            as above. Default ["random"] 
        methods (List[str]): List of UQ methods to use (e.g.,
            "evidence_new_reg", "evidence", "dropout", "ensemble"). 
            Default "evidence_new_reg".
        reg_coefs (List[float]): List of floats to use when you running
            evidence method. Note; This must be the same size as the methods
            list for convenience
        split_sizes (List[float}): List of train, val, test split sizes for this trial to run. 
            Default [0.8, 0.1, 0.1] 
        save_dir (str): Save directory. Default: "results/submission_results_paper"
        dropout (float): Amount of dropout for dropout uncrtainty method.
            Default 0.1
        error (str): Type of error to run with. Deafult "rmse"
        debug (bool): If true, debug. Default False
        use_gpu (bool): If true, run wtith gres. Default True
        job_name_prefix (str): Name of log file outputs. Default "gnn_job"
        experiment_name (str): Name to launch job on clust. Default "chemprop"
        num_minutes (int): Num minutes for the job on the cluster. Default 30
        num_hours (int): Num hours for the job on the cluster. Default 1
        num_days (int): Num days for the job on the cluster. Default 0
        use_slurm (bool): If true, use slurm launcher. Default True. 
        atomistic (bool): If true, launch atomistic experiments. 
        experiment_file_name (Optional[str]): Name of the expriment file used to launch
            this
        max_lr_atomistic (float): Max learning rate ofr atomistic. 
            Default 0.0002.
        no_smiles_export (bool): If true, don't export smiles. Default False.
        final_lr_atomistic (float): Final lr atomistic. Default 1e-4.
    """ 

    # Verify the length constraints
    assert (len(datasets) == len(dataset_types)) 
    assert (len(datasets) == len(dataset_splits)) 
    assert (len(dataset_types) == len(dataset_splits))
    assert (len(methods) == len(reg_coefs)) 
    assert (len(split_sizes) == 3)  

    # Dump out file
    if experiment_file_name: 
        dump_config_file(save_dir, experiment_file_name)


    # Handle sbatch args
    SBATCH_ARGS = f"--output=logs/{job_name_prefix}_%J.log -J {experiment_name} -t {num_days}-{num_hours}:{num_minutes}:00"
    if use_gpu: 
        SBATCH_ARGS = f"--gres=gpu:volta:1 {SBATCH_ARGS}" 

    BASE_ARGS = "--save_confidence conf.txt --use_entropy --confidence_evaluation_methods cutoff" 

    if no_smiles_export: 
        BASE_ARGS = f"{BASE_ARGS} --no_smiles_export"

    if atomistic: 
        SBATCH_ARGS = f"{SBATCH_ARGS} -c 2"
        if use_slurm: BASE_ARGS = f"{BASE_ARGS} --slurm_job"


    for trial in seeds:
        seed = trial
        for dataset, dataset_type, split_type in zip(datasets, dataset_types, dataset_splits): 

            save_dir_ = os.path.join(f"{save_dir}", split_type)
            for coeff, method in zip(reg_coefs, methods): 
                METHOD_ARGS = ""
                method_name = method
                if method == "ensemble": 
                    METHOD_ARGS=f"--ensemble_size {n_ensembles} --threads {ensemble_threads}"
                elif method == "dropout": 
                    METHOD_ARGS=f"--ensemble_size {n_ensembles} --dropout {dropout} --no_dropout_inference"
                elif method == "evidence": 
                    raise ValueError("Should only see new evidence with evidence_new_reg")
                elif method == "evidence_new": 
                    raise ValueError("Should only see new evidence with evidence_new_reg")
                    METHOD_ARGS=f"--new_loss"
                    method_name= "evidence"
                elif method == "evidence_new_reg": 
                    METHOD_ARGS=f"--new_loss --regularizer_coeff {coeff}"
                    method_name= "evidence"
                    # Rename method to have a name for the coefficient
                    method = f"{method}_{coeff}"
                else: 
                    pass

                SPLIT_ARGS=f"--split_type {split_type} --split_sizes {split_sizes[0]} {split_sizes[1]} {split_sizes[2]}" 

                # Add train split here
                LOG_ARGS = f"--save_dir {save_dir_}/{dataset}/{method}" 
                
                # Set dataset args
                if dataset == "freesolv": 
                    EPOCH_ARGS=f"--epochs {epochs}"
                elif dataset=="qm9": 
                    EPOCH_ARGS=f"--epochs {epochs} --metric mae"
                elif dataset=="qm7": 
                    EPOCH_ARGS=f"--epochs {epochs} --metric rmse"
                elif dataset=="delaney": 
                    EPOCH_ARGS= f"--epochs {epochs}"
                else:
                    EPOCH_ARGS=f"--epochs {epochs}"
                
                if debug: 
                    EPOCH_ARGS="--epochs 3"

                if atomistic: 

                    # assert atomistic args
                    assert (dataset == "qm9")
                    assert (dataset_type == "regression")
                    assert (split_type == "random")
                    assert (method != "dropout")

                    LEARNING_ARGS = f"--batch_size 100 --final_lr {final_lr_atomistic} --max_lr {max_lr_atomistic}" #--patience 20" --max_lr 0.0002"

                    python_string=f"python train_atomistic.py --atomistic {LEARNING_ARGS}  --confidence {method_name} {EPOCH_ARGS} {METHOD_ARGS} {LOG_ARGS} {BASE_ARGS} {SPLIT_ARGS} --seed {trial} --dataset_type {dataset_type} --data_path data/{dataset}.db --atomistic" 

                else: 
                    python_string=f"python train.py --confidence {method_name} {EPOCH_ARGS} {METHOD_ARGS} {LOG_ARGS} {BASE_ARGS} {SPLIT_ARGS} --seed {trial} --dataset_type {dataset_type} --data_path data/{dataset}.csv"

                if debug:
                    python_string = f"{python_string} --debug"

                bash_string=f"sbatch {SBATCH_ARGS} --export=CMD=\"{python_string}\" scripts/generic_slurm.sh"
    
                if use_slurm: 
                    print(f"{bash_string}")
                    subprocess.call(bash_string, shell=True)
                else: 
                    print(f"{python_string}")
                    subprocess.call(python_string, shell=True)

                time.sleep(3)

if __name__ == "__main__": 
    os.makedirs("logs", exist_ok=True)
    args, exp_file  = get_args()
    main( experiment_file_name = exp_file, **args)



