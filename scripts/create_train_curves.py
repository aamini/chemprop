"""Create train curve from log file

Call signatures used: 

python scripts/create_train_curves.py --log-dir submission_results/gnn/qm9/evidence/

# Note : for this case, it's worth rescaling the x axis
python scripts/create_train_curves.py --log-dir submission_results_atomsitic_multi/gnn/qm9/evidence/ --verbose-extension fold_0/model_0/verbose.log



"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 
import argparse
import re
import pandas as pd


def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", help="name of log file") 
    parser.add_argument("--log-dir", help="name of log dir (eg has diff trials)") 
    parser.add_argument("--verbose-extension", 
                        help="path from the trial directories to the log",
                        default="verbose.log")  #fold_0/model_0/verbose.log
    parser.add_argument("--out-name", 
                        help="Output name", 
                        default="temp.png")  #fold_0/model_0/verbose.log

    return parser.parse_args()


def get_val_losses(log_file : str): 
    """ Extract the validation epochs from the log file"""
    losses = []
    ctr = 0 
    epoch_re = r"Validation mae = (\d+\.\d+) *\nEpoch (\d+)"
    lines = open(log_file, "r").readlines()
    for index, line in enumerate(lines[:-1]): 
        ctr += 1
        search_str = "".join([line,lines[index+1]])
        examples = re.findall(epoch_re, search_str)
        if len(examples) > 0: 
            loss, epoch = examples[0]
            losses.append(float(loss))
    return losses

if __name__=="__main__": 
    args = get_args()
    log_dir = args.log_dir
    log_file= args.log_file
    if log_dir: 
        trial_files = [os.path.join(log_dir, i) for i in os.listdir(log_dir)]
        epoch_loss_list = [] 
        for log_file in [f"{j}/{args.verbose_extension}" for j in trial_files]: 
            if os.path.isfile(log_file): 
                print(log_file)
                epoch_losses = get_val_losses(log_file)
                for index, loss in enumerate(epoch_losses): 
                    loss_entry = {"epoch" : index, "loss" : loss}
                    epoch_loss_list.append(loss_entry)
        df = pd.DataFrame(epoch_loss_list)
        g = sns.lineplot(data =df, x="epoch", y="loss")
        #plt.ylim([0,0.5])
        plt.savefig(f"{args.out_name}")


    else:
        epoch_losses = get_val_losses(log_file)






