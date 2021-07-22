#!/bin/bash
#SBATCH --job-name="combine_confs"
#SBATCH -t 0-05:00:00   # 5 hours


source /etc/profile 
source /home/gridsan/samlg/.bashrc
conda activate chemprop

# Classif
srun python scripts/combine_conf_outputs.py --dataset-dir results/2021_01_31_combined_logs/2020_12_21_classif/random/ --outfile results/2021_01_31_combined_logs/2020_12_21_classif/random/consolidated_classif --result-type classif

# Low n 
srun python scripts/combine_conf_outputs.py --dataset-dir results/2021_01_31_combined_logs/2021_01_31_low_n/random/ --outfile results/2021_01_31_combined_logs/2021_01_31_low_n/random/consolidated_low_n --result-type low_n

# High N
srun python scripts/combine_conf_outputs.py --dataset-dir results/2021_01_31_combined_logs/2020_12_05_high_n/random/ --outfile results/2021_01_31_combined_logs/2020_12_05_high_n/random/consolidated_high_n --result-type high_n

# Atomistic
srun python scripts/combine_conf_outputs.py --dataset-dir results/2021_01_31_combined_logs/2021_01_04_atomistic/random/ --outfile results/2021_01_31_combined_logs/2021_01_04_atomistic/random/consolidated_atomistic --result-type atomistic

# Tdc
srun python scripts/combine_conf_outputs.py --dataset-dir results/2021_01_31_combined_logs/2021_06_20_tdc/random --outfile results/2021_01_31_combined_logs/2021_06_20_tdc/random/consolidated_tdc --result-type tdc
