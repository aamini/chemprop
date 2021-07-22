#!/bin/bash
#SBATCH --job-name="combine_confs"
#SBATCH -t 0-05:00:00   # 5 hours


source /etc/profile 
source /home/gridsan/samlg/.bashrc
conda activate chemprop


# Classif
srun python scripts/make_figs.py --full-data-file  results/2021_01_31_combined_logs/2020_12_21_classif/random/consolidated_classif.tsv --summary-data-file results/2021_01_31_combined_logs/2020_12_21_classif/random/consolidated_classif_summary_calc.tsv --plot-type classif --outdir results/2021_01_31_combined_logs/figures

# Low N
srun python scripts/make_figs.py --full-data-file  results/2021_01_31_combined_logs/2021_01_31_low_n/random/consolidated_low_n.tsv --summary-data-file results/2021_01_31_combined_logs/2021_01_31_low_n/random/consolidated_low_n_summary_calc.tsv --plot-type low_n --outdir results/2021_01_31_combined_logs/figures

# TDC
srun python scripts/make_figs.py --full-data-file  results/2021_01_31_combined_logs/2021_06_20_tdc/random/consolidated_tdc.tsv --summary-data-file results/2021_01_31_combined_logs/2021_06_20_tdc/random/consolidated_tdc_summary_calc.tsv --plot-type tdc --outdir results/2021_01_31_combined_logs/figures

# High N
srun python scripts/make_figs.py --full-data-file  results/2021_01_31_combined_logs/2020_12_05_high_n/random/consolidated_high_n.tsv  --summary-data-file results/2021_01_31_combined_logs/2020_12_05_high_n/random/consolidated_high_n_summary_calc.tsv --plot-type high_n --outdir results/2021_01_31_combined_logs/figures

# Atomistic
srun python scripts/make_figs.py --full-data-file  results/2021_01_31_combined_logs/2021_01_04_atomistic/random/consolidated_atomistic.tsv  --summary-data-file results/2021_01_31_combined_logs/2021_01_04_atomistic/random/consolidated_atomistic_summary_calc.tsv --plot-type atomistic --outdir results/2021_01_31_combined_logs/figures



