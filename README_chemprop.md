# Evidential Deep Learning for Guided Molecular Property Prediction and Discovery

The repository contains all of the code and instructions needed to reproduce the experiments and results of **Evidential Deep Learning for Guided Molecular Property Prediction and Discovery**. We detail the changes we have made in the [Evidential Uncertainty](#evidential-uncertainty) section of this README. We acknowledge the original [Chemprop repository](https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.0c00502?casa_token=dUy7JEOj9Y4AAAAA:8s14xrMIr020lqI3mFF8t-mG_U4TtaCd1Kv-3ECkksQZUnzS5uAiKi2qg1xFUTMQonDnmuzOB2Qrsdw) which this code leveraged and built on top of.


## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  * [Option 1: Conda](#option-1-conda)
  * [Option 2: Docker](#option-2-docker)
  * [(Optional) Installing `chemprop` as a Package](#optional-installing-chemprop-as-a-package)
  * [Notes](#notes)
- [Web Interface](#web-interface)
- [Data](#data)
- [Training](#training)
  * [Train/Validation/Test Splits](#train-validation-test-splits)
  * [Cross validation](#cross-validation)
  * [Ensembling](#ensembling)
  * [Hyperparameter Optimization](#hyperparameter-optimization)
  * [Additional Features](#additional-features)
    * [RDKit 2D Features](#rdkit-2d-features)
    * [Custom Features](#custom-features)
- [Evidential Uncertainty](#evidential-uncertainty)
  * [Training a Model](#training-a-model)
  * [Evidence Comparison Demo](#evidence-comparison-demo)
  * [Benchmarking](#benchmarking)
  * [Active Learning](#active-learning)
  * [Stokes Analysis](#stokes-analysis)
- [Predicting](#predicting)
- [TensorBoard](#tensorboard)
- [Results](#results)

## Requirements

While it is possible to run all of the code on a CPU-only machine, GPUs make training significantly faster. To run with GPUs, you will need:
 * cuda >= 8.0
 * cuDNN

## Installation

### Option 1: Conda

The easiest way to install the `chemprop` dependencies is via conda. Here are the steps:

1. Install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
2. `cd /path/to/chemprop`
3. `conda env create -f environment.yml`
4. `source activate chemprop` (or `conda activate chemprop` for newer versions of conda)
5. (Optional) `pip install git+https://github.com/bp-kelley/descriptastorus`

The optional `descriptastorus` package is only necessary if you plan to incorporate computed RDKit features into your model (see [Additional Features](#additional-features)). The addition of these features improves model performance on some datasets but is not necessary for the base model.

### Option 2: Docker

Docker provides a nice way to isolate the `chemprop` code and environment. To install and run our code in a Docker container, follow these steps:

1. Install Docker from [https://docs.docker.com/install/](https://docs.docker.com/install/)
2. `cd /path/to/chemprop`
3. `docker build -t chemprop .`
4. `docker run -it chemprop:latest /bin/bash`

Note that you will need to run the latter command with nvidia-docker if you are on a GPU machine in order to be able to access the GPUs.

### (Optional) Installing `chemprop` as a Package

If you would like to use functions or classes from `chemprop` in your own code, you can install `chemprop` as a pip package as follows:

1. `cd /path/to/chemprop`
2. `pip install .`

Then you can use `import chemprop` or `from chemprop import ...` in your other code.

### Notes

If you get warning messages about `kyotocabinet` not being installed, it's safe to ignore them.

## Web Interface

For those less familiar with the command line, we also have a web interface which allows for basic training and predicting. After installing the dependencies following the instructions above, you can start the web interface by running `python web/run.py` and then navigating to [localhost:5000](http://localhost:5000) in a web browser.

![Training with our web interface](web/app/static/images/web_train.png "Training with our web interface")

![Predicting with our web interface](web/app/static/images/web_predict.png "Predicting with our web interface")


## Data

In order to train a model, you must provide training data containing molecules (as SMILES strings) and known target values. Targets can either be real numbers, if performing regression, or binary (i.e. 0s and 1s), if performing classification. Target values which are unknown can be left as blanks.

Our model can either train on a single target ("single tasking") or on multiple targets simultaneously ("multi-tasking").

The data file must be be a **CSV file with a header row**. For example:
```
smiles,NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53
CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0,0,1,,,0,0,1,0,0,0,0
CCN1C(=O)NC(c2ccccc2)C1=O,0,0,0,0,0,0,0,,0,,0,0
...
```
Datasets from [MoleculeNet](http://moleculenet.ai/) and a 450K subset of ChEMBL from [http://www.bioinf.jku.at/research/lsc/index.html](http://www.bioinf.jku.at/research/lsc/index.html) have been preprocessed and are available in `data.tar.gz`. To uncompress them, run `tar xvzf data.tar.gz`.

## Training

To train a model, run:
```
python train.py --data_path <path> --dataset_type <type> --save_dir <dir>
```
where `<path>` is the path to a CSV file containing a dataset, `<type>` is either "classification" or "regression" depending on the type of the dataset, and `<dir>` is the directory where model checkpoints will be saved.

For example:
```
python train.py --data_path data/tox21.csv --dataset_type classification --save_dir tox21_checkpoints
```

Notes:
* The default metric for classification is AUC and the default metric for regression is RMSE. Other metrics may be specified with `--metric <metric>`.
* `--save_dir` may be left out if you don't want to save model checkpoints.
* `--quiet` can be added to reduce the amount of debugging information printed to the console. Both a quiet and verbose version of the logs are saved in the `save_dir`.

### Train/Validation/Test Splits

Our code supports several methods of splitting data into train, validation, and test sets.

**Random:** By default, the data will be split randomly into train, validation, and test sets.

**Scaffold:** Alternatively, the data can be split by molecular scaffold so that the same scaffold never appears in more than one split. This can be specified by adding `--split_type scaffold_balanced`.

**Separate val/test:** If you have separate data files you would like to use as the validation or test set, you can specify them with `--separate_val_path <val_path>` and/or `--separate_test_path <test_path>`.

Note: By default, both random and scaffold split the data into 80% train, 10% validation, and 10% test. This can be changed with `--split_sizes <train_frac> <val_frac> <test_frac>`. For example, the default setting is `--split_sizes 0.8 0.1 0.1`. Both also involve a random component and can be seeded with `--seed <seed>`. The default setting is `--seed 0`.

### Cross validation

k-fold cross-validation can be run by specifying `--num_folds <k>`. The default is `--num_folds 1`.

### Ensembling

To train an ensemble, specify the number of models in the ensemble with `--ensemble_size <n>`. The default is `--ensemble_size 1`.

### Hyperparameter Optimization

Although the default message passing architecture works quite well on a variety of datasets, optimizing the hyperparameters for a particular dataset often leads to marked improvement in predictive performance. We have automated hyperparameter optimization via Bayesian optimization (using the [hyperopt](https://github.com/hyperopt/hyperopt) package) in `hyperparameter_optimization.py`. This script finds the optimal hidden size, depth, dropout, and number of feed-forward layers for our model. Optimization can be run as follows:
```
python hyperparameter_optimization.py --data_path <data_path> --dataset_type <type> --num_iters <n> --config_save_path <config_path>
```
where `<n>` is the number of hyperparameter settings to try and `<config_path>` is the path to a `.json` file where the optimal hyperparameters will be saved. Once hyperparameter optimization is complete, the optimal hyperparameters can be applied during training by specifying the config path as follows:
```
python train.py --data_path <data_path> --dataset_type <type> --config_path <config_path>
```

### Additional Features

While the model works very well on its own, especially after hyperparameter optimization, we have seen that adding computed molecule-level features can further improve performance on certain datasets. Features can be added to the model using the `--features_generator <generator>` flag.

#### RDKit 2D Features

As a starting point, we recommend using pre-normalized RDKit features by using the `--features_generator rdkit_2d_normalized --no_features_scaling` flags. In general, we recommend NOT using the `--no_features_scaling` flag (i.e. allow the code to automatically perform feature scaling), but in the case of `rdkit_2d_normalized`, those features have been pre-normalized and don't require further scaling.

Note: In order to use the `rdkit_2d_normalized` features, you must have `descriptastorus` installed. If you installed via conda, you can install `descriptastorus` by running `pip install git+https://github.com/bp-kelley/descriptastorus`. If you installed via Docker, `descriptastorus` should already be installed.

#### Custom Features

If you would like to load custom features, you can do so in two ways:

1. **Generate features:** If you want to generate features in code, you can write a custom features generator function in `chemprop/features/features_generators.py`. Scroll down to the bottom of that file to see a features generator code template.
2. **Load features:** If you have features saved as a numpy `.npy` file or as a `.csv` file, you can load the features by using `--features_path /path/to/features`. Note that the features must be in the same order as the SMILES strings in your data file. Also note that `.csv` files must have a header row and the features should be comma-separated with one line per molecule.

## Evidential Uncertainty

 This fork implements message passing neural networks with [deep evidential regression](https://arxiv.org/pdf/1910.02600.pdf). The changes made to implement evidential uncertainty can be found in:

1. **Model modifications:** Because the evidential model requires outputting 4 auxilary parameters of the evidential distribution for every single desired target, we have modified `chemprop/model/model.py`.
2. **Predicting uncertainty:** The predict module in `chemprop/train/predict.py` has been modified to convert these four parmaeters into an uncertainty prediction.
3. **Loss function:** The evidential loss function can be found in `chemprop/utils.py`. We note that the most up to date version of this loss function is accessed using the `--new_loss` flag.
4. **Addition of SchNet support:** In order to demonstrate evidential uncertianty on SchNet, we have integrated the SchNet package into the Chemprop training procedure by modifying `chemprop/model/model.py` and `chemprop/train/run_training.py`. `train_atomistic.py` serves as an entrypoint to use this model. Demonstrations for benchmarking with this model can be fund in the [Benchmarking](#benchmarking) section.

#### Training a Model

To train a method with evidential regression on a dataset, the following command can be run:

`python train.py --confidence evidence --epochs 20 --new_loss --regularizer_coeff 0.2 --save_dir results/evidence_demo --save_confidence conf.txt --confidence_evaluation_methods cutoff --split_type random --split_sizes 0.8 0.1 0.1 --seed 0 --dataset_type regression --data_path data/delaney.csv`

This will run 20 epochs of training an evidence regression model on the delaney dataset, saving the results of the run in the conf.txt file.

Note: The new\_loss flag is necessary to use the most up to date version of the evidential loss, which was used to produce all results in our paper.

#### Evidence Comparison Demo

For a simple demonstration of the pipeline to reproduce benchmarking plots, we've prepared a configuration file to run 2 trials of dropout, ensembles, and evidential regression for an abridged 20 epochs of training on the small, Delaney dataset, which can be be completed in under 20 minutes on a CPU. This can be run using the command:

`python scripts/run_benchmarking.py configs/reproducibility/delaney_demo.json`

To consolidate the results from this demo:

`python scripts/combine_conf_outputs.py --dataset-dir results/delaney_demo/random  --outfile results/delaney_demo/consolidated_delaney_demo --result-type low_n`

To produce figures from the results of this demo:

`python scripts/make_figs.py --full-data-file  results/delaney_demo/consolidated_delaney_demo.tsv --summary-data-file results/delaney_demo/consolidated_delaney_demo_summary_calc.tsv --plot-type low_n --outdir results/delaney_demo/figures`

This will generate figures in the folder `results/delaney_demo/figures/low_n`. Please note that all tuning plots will show no results, as this demonstration uses only a single evidence coefficient.

#### Benchmarking

To reproduce all benchmarking plots, we provide the following config files, which are currently configured to use a SLURM scheduler for parallelization and take advantage of GPU support.

1. **Generating Benchmarking:** The following config files can all be run using the command `python scripts/run_benchmarking.py [config_name]`  
    * `configs/reproducibility/low_n_config.json`  
    * `configs/reproducibility/high_n_config.json`  
    * `configs/reproducibility/atomistic_config.json`  
    * `configs/reproducibility/classif_config.json`  

These configs will each generate logs for the different trials respectively. The low N config will conduct a sweep over various evidential loss coefficient weights in order to generate the corresponding tuning plots. The config file can be modified by setting `use_slurm` to false in order to avoid using the SLURM launcher.

2. **Combining Results:** Results from the different benchmarking trials can be combined using the various labeled commands in the script `scripts/combine_benchmarking.sh`. To avoid using SLURM, these can be individually copied and run without the `srun` prefix.

3. **Generating figures:** The figures for the different dataset types can be generated with commands listed in the script `make_all_figs.sh`. Once again, this takes advantage of the SLURM launcher, but can be used directly by removing the `srun` flags.

#### Active Learning

To execute an active learning experiment on the QM9 dataset, as shown in the paper, variants of the following command can be executed. The commands in the following examples will launch a signle trial of an active learning experiment, in which an explorative acquisition strategy, i.e., wherein data points with the greatest uncertainties are prioritized for selection, is compared to a random acquisition strategy. The `active_learning.py` script contains the code to execute these experiments. Multiple trials can be parallelized to generate performance statistics. For example,

`python active_learning.py --confidence evidence --new_loss regularizer_coeff 0.2 --batch_size 50 --epochs 100 --num_al_loops 7 --al_init_ratio 0.15 --al_strategy explorative --data_path ./data/qm9.csv --dataset_type regression --save_dir logs/qm9/evidence --save_confidence conf.txt --split_type random --use_std --use_entropy --quiet`

will execute two complete active learning loops starting from 15\% of the entire training set and iteratively acquiring 7 batches of data based on an explorative evidence-based strategy. `--al_strategy` can be switch to `random` to run baselines against a random acquisition strategy. Once trained, logs can be passed into `scripts/plot_al_results_cutoff.py` to reproduce the relevant plots from the manuscript.

This execution will yield `.csv` files containing performance metrics (i.e., RMSE) as a function of training set size, with the acquisition strategy delineated. To plot the results of an active learning experiment, the following command can be run:

`python scripts/plot_al_results_cutoff.py --path [/path/to/logs ...]`

where `[/path/to/logs ...]` should have a sub-directory of the form `[dataset]/[method]/scores`, where dataset is the specified dataset, i.e., QM9.

#### Stokes Analysis

The presented analysis on the datasets from [Stokes et al. Cell 2020](https://www.sciencedirect.com/science/article/abs/pii/S0092867420301021?via%3Dihub) can be summarized as follows:

1. **Train:** An evidential D-MPNN is trained on the primary dataset reported by Stokes et al. of small molecules and their *in vitro* antibiotic activity against *Escherichia coli*, measured as the optical density at a wavelength of 600 nm (OD600), where smaller OD600 values correspond to greater antibiotic activity of the associated small molecule. We train an evidential D-MPNN for regression, where the target is the OD600 for a particular small molecule input.
2. **Predict:** The trained evidential D-MPNN is then evaluated on an independent test set, the Broad Institute's [Drug Repurposing Hub](https://www.broadinstitute.org/drug-repurposing-hub). The model predicts a continuous target, predicted OD600, directly from a SMILES string molecule input, as well as an associated evidential uncertainty. For a subset of molecules from the Drug Repurposing Hub, there exist experimental measurements of their true *in vitro* antibiotic activity against *E. coli*, as reported by Stokes et al. The subset of molecules for which experimental annotations are available are the focus of the analysis in (3).
3. **Prioritize:** The top-k (k=50 for the results in the paper) ranking predictions from (2) are considered, and confidence filters, determined by the distribution of predicted evidential uncertainties from (2), are applied to filter the list of candidates. The experimental hit rate is determined by calculating the overlap between the filtered list of candidates and those repurposing molecules experimentally measured to result in an OD600 < 0.2 in the real world (active antibiotic).

The analysis presented in the paper uses a D-MPNN based on that from Stokes et al., which includes augmentation with pre-computed features from RDKit. First to compute the features, the [`descriptastorus`](https://anaconda.org/RMG/descriptastorus) must be installed. To compute the features for both the Stokes primary data and the Broad repurposing library:

`
python scripts/save_features.py --features_generator rdkit_2d_normalized --data_path data/stokes_primary_regr.csv --save_path features/stokes_primary_regr
`

`
python scripts/save_features.py --features_generator rdkit_2d_normalized --data_path data/broad_smiles_validated_full.csv --save_path features/stokes_broad
`

**Step 1: Train.** This will train an evidential regression D-MPNN on the Stokes primary dataset. Data are sampled to adjust for label imbalance according to the OD600 < 0.2 cutoff. Experiments are run with 30 epochs, 10 fold cross-validation, 20 independent trials with varying random seeds, data balancing probability of 0.1, and evidential regularization coefficient of 0.1.

`
python train.py --data_path=data/stokes_primary_regr.csv  --dataset_type=regression --split_type=random --save_smiles_splits --save_confidence conf.txt --features_path features/stokes_primary_regr.npz --seed [RANDOM] --no_features_scaling --num_folds 10 --confidence=evidence --new_loss --epochs 30 --regularizer_coeff 0.1 --stokes_balance 0.1 save_dir=logs/stokes/evidence_30_0.1_0.1
`

**Step 2: Predict.** This will utilize the trained evidential regression D-MPNN for prediction of antibiotic activity and uncertainty on the Broad Repurposing Hub dataset.

`
python predict.py --test_path data/broad_smiles_validated_full.csv --features_path features/stokes_broad.npz --preds_path logs/stokes/evidence_30_0.1_0.1/broad/predictions_[RANDOM].csv --checkpoint_dir logs/stokes/evidence_30_0.1_0.1/[checkpoint]
`

**Step 3: Prioritize & Evaluate.** This will take the resulting predictions and perform the confidence-based prioritization described above to reproduce the plots shown.

`
python scripts/stokes_figures.py --use_stds --preds_path logs/stokes --method evidence_30_0.1_0.1
`

## Predicting

To load a trained model and make predictions, run `predict.py` and specify:
* `--test_path <path>` Path to the data to predict on.
* A checkpoint by using either:
  * `--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. `--save_dir` during training). This will walk the directory, load all `.pt` files it finds, and treat the models as an ensemble.
  * `--checkpoint_path <path>` Path to a model checkpoint file (`.pt` file).
* `--preds_path` Path where a CSV file containing the predictions will be saved.

For example:
```
python predict.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints --preds_path tox21_preds.csv
```
or
```
python predict.py --test_path data/tox21.csv --checkpoint_path tox21_checkpoints/fold_0/model_0/model.pt --preds_path tox21_preds.csv
```

## TensorBoard

During training, TensorBoard logs are automatically saved to the same directory as the model checkpoints. To view TensorBoard logs, run `tensorboard --logdir=<dir>` where `<dir>` is the path to the checkpoint directory. Then navigate to [http://localhost:6006](http://localhost:6006).

## Results

We compared our model against MolNet by Wu et al. on all of the MolNet datasets for which we could reproduce their splits (all but Bace, Toxcast, and qm7). When there was only one fold provided (scaffold split for BBBP and HIV), we ran our model multiple times and reported average performance. In each case we optimize hyperparameters on separate folds, use rdkit_2d_normalized features when useful, and compare to the best-performing model in MolNet as reported by Wu et al. We did not ensemble our model in these results.

Results on classification datasets (AUC score, the higher the better)

| Dataset | Size |	Ours |	MolNet Best Model |
| :---: | :---: | :---: | :---: |
| BBBP | 2,039 | 0.735 ± 0.0064	| 0.729 |
| Tox21 | 7,831 | 0.855 ± 0.0052	| 0.829 ± 0.006 |
| Sider | 1,427 |	0.678 ± 0.019	| 0.648 ± 0.009 |
| clintox | 1,478 | 0.9 ± 0.0089	| 0.832 ± 0.037 |
| MUV | 93,087 | 0.0897 ± 0.015 | 0.184 ± 0.02 |
| HIV | 41,127 |	0.793 ± 0.0012 |	0.792 |
| PCBA | 437,928 | 0.397 ± .00075 | 	0.136 ± 0.004 |

Results on regression datasets (score, the lower the better)

Dataset | Size | Ours | GraphConv/MPNN (deepchem) |
| :---: | :---: | :---: | :---: |
delaney	| 1,128 | 0.567 ± 0.026 | 0.58 ± 0.03 |
Freesolv | 642 |	1.11 ± 0.035 | 1.15 ± 0.12 |
Lipo | 4,200 |	0.542 ± 0.02 |	0.655 ± 0.036 |
qm8 | 21,786 |	0.0082 ± 0.00019 | 0.0143 ± 0.0011 |
qm9 | 133,884 |	2.03 ± 0.021	| 2.4 ± 1.1 |

Lastly, you can find the code to our original repo at https://github.com/wengong-jin/chemprop and for the Mayr et al. baseline at https://github.com/yangkevin2/lsc_experiments .
