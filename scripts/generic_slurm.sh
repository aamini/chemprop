#!/bin/bash
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH -n 1           # 1 core
#SBATCH -t 0-05:00:00   # 5 hours
#SBATCH -J atomistic # sensible name for the job
#SBATCH --output=logs/slurm_generic_%j.log   # Standard output and error log
##SBATCH --gres=gpu:volta:1 #1 gpu

##SBATCH --mail-user=samlg@mit.edu # mail to me
##SBATCH --mem=20000  # 20 gb 
##SBATCH -p {Partition Name} # Partition with GPUs

# Use this to run generic scripts:
# sbatch --export=CMD="python my_python_script --my-arg" src/scripts/slurm_scripts/generic_slurm.sh


# Import module
source /etc/profile 

# Load some modules
module load anaconda/2020b
module load cuda/10.1

source /home/gridsan/samlg/.bashrc

# module load c3ddb/glibc/2.14
# module load cuda80/toolkit/8.0.44

# Activate conda
# source {path}/miniconda3/etc/profile.d/conda.sh

# Activate right python version
# conda activate {conda_env}
conda activate chemprop

# Evaluate the passed in command... in this case, it should be python ... 
eval $CMD

