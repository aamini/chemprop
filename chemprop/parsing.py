from argparse import ArgumentParser, Namespace
import json
import os
from tempfile import TemporaryDirectory
from datetime import datetime
import torch

from chemprop.utils import makedirs
from chemprop.features import get_available_features_generators


def add_predict_args(parser: ArgumentParser):
    """
    Adds predict arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--test_path', type=str,
                        help='Path to CSV file containing testing data for which predictions will be made')
    parser.add_argument('--compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--preds_path', type=str,
                        help='Path to CSV file where predictions will be saved')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')


def add_atomistic_args(parser: ArgumentParser):
    """
    Adds schnet  arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--n_atom_basis', type=int, default=128,
                        help='Number of atoms to use for representation')
    parser.add_argument('--n_filters', type=int, default=128,
                        help='Number of filters to use')
    parser.add_argument('--n_interactions', type=int, default=6,
                        help='Number of interaction layers to use')
    parser.add_argument('--n_gaussians', type=int, default=50,
                        help='Number of Gaussians to use')
    parser.add_argument('--cutoff', type=int, default=10,
                        help='Cosine function cutoff')
    parser.add_argument('--slurm_job', default=False,
                        action="store_true",
                        help='If true, locally copy qm9db')

    # Scheduler params
    parser.add_argument('--patience', default=25,
                        action="store", type=int,
                        help="Epochs to wait before decreasing lr")
    parser.add_argument('--factor', default=0.8,
                        action="store", type=float,
                        help='Factor to decrease LR by')

def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--debug', action="store_true",
                        default=False,
                        help='If true, subset the data for debugging')
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Whether to skip training and only test the model')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network')
    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test splits for prediction convenience later')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression'],
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.')
    parser.add_argument('--separate_val_path', type=str,
                        help='Path to separate val set, optional')
    parser.add_argument('--separate_val_features_path', type=str, nargs='*',
                        help='Path to file with features for separate val set')
    parser.add_argument('--separate_test_path', type=str,
                        help='Path to separate test set, optional')
    parser.add_argument('--separate_test_features_path', type=str, nargs='*',
                        help='Path to file with features for separate test set')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'scaffold',
                                 'predetermined', 'ood_test'],
                        help='Method of splitting the data into train/val/test')

    parser.add_argument('--ood_save_dir', type=str,
                        default="data/saved_dist",
                        help='Directory save all by all tani sim for dataset')

    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_folds', type=int, default=1,
                        help='Number of folds when performing cross validation')
    parser.add_argument('--folds_file', type=str, default=None,
                        help='Optional file of fold labels')
    parser.add_argument('--val_fold_index', type=int, default=None,
                        help='Which fold to use as val for leave-one-out cross val')
    parser.add_argument('--test_fold_index', type=int, default=None,
                        help='Which fold to use as test for leave-one-out cross val')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_folds` > 1, the first fold uses this seed and all'
                             'subsequent folds add 1 to the seed.')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['auc', 'prc-auc', 'rmse',
                                 'mae', 'r2', 'accuracy'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).'
                             'Note: Defaults to "auc" for classification and "rmse" for regression.')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='The number of batches between each logging of the training loss')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--show_individual_scores', action='store_true', default=False,
                        help='Show all scores for individual targets, not just average, at the end')
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--config_path', type=str,
                        help='Path to a .json file containing arguments. Any arguments present in the config'
                             'file will override arguments specified via the command line or by the defaults.')
    parser.add_argument('--atomistic', action="store_true", default=False,
                        help='If true, use atomistic networks in simulation')
    parser.add_argument('--task_inds', type=int, default=[], nargs='+',
                        help='Indices of tasks you want to train on.')
    parser.add_argument('--test_preds_path', type=str,
                        help='Path to where predictions on test set will be saved.')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')
    parser.add_argument('--stokes_balance', type=float, default=1,
                        help='Balance for Stokes analysis')

    # Model arguments
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of parallel threads to spawn ensembles in. 1 thread trains in serial.')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU',
                                 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Undirected edges (always sum the two relevant bond vectors)')
    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='Use messages on atoms instead of messages on bonds')

    # Confidence Arguments
    parser.add_argument('--confidence', type=str,
                        choices=[None, 'gaussian', 'random_forest', 'ensemble',
                                 'tanimoto', 'conformal', 'probability',
                                 'conformal', 'nn', 'boost', 'latent_space',
                                 'bootstrap', 'snapshot', 'dropout',
                                 'fp_random_forest', 'fp_gaussian', 'evidence',
                                 'sigmoid'], default=None,
                        help='Measure confidence values for the prediction.')
    parser.add_argument("--regularizer_coeff", type=float,
                        default=1.0,
                        help="Coefficient to scale the loss function regularizer")
    parser.add_argument('--new_loss', action="store_true", default=False,
                        help=("If true, use the new evidence loss"
                              " prediction with the model"))
    parser.add_argument('--no_dropout_inference', action="store_true", default=False,
                        help="If true, don't use dropout for mean inference")

    parser.add_argument('--use_entropy', action="store_true", default=False,
                        help=("If true, also output the entropy for each"
                              " prediction with the model"))

    parser.add_argument('--no_smiles_export', action="store_true", default=False,
                        help=("If set, avoid storing the smiles with exported"
                              "data prediction with the model"))

    parser.add_argument('--calibrate_confidence', action='store_true', default=False, help='Calibrate confidence by test data.')
    parser.add_argument('--save_confidence', type=str, default=None,
                        help='Measure confidence values for the prediction.')
    parser.add_argument('--last_hidden_size', type=int, default=300,
                        help='Size of last hidden layer.')
    parser.add_argument('--confidence_evaluation_methods',
                        type=str,
                        default=[],
                        nargs='+',
                        help='List of confidence evaluation methods.')

    # Active Learning Arguments
    parser.add_argument('--al_init_ratio', type=float, default=0.1,
                        help='Percent of training data to use on first active learning iteration')
    parser.add_argument('--al_end_ratio', type=float, default=None,
                        help='Fraction of total data To stop active learning early. By default, explore full train data')
    parser.add_argument('--num_al_loops', type=int, default=20,
                        help='Number of active learning loops to add new data')
    parser.add_argument('--al_topk', type=int, default=1000,
                        help='Top-K acquired molecules to consider during active learning')

    parser.add_argument('--al_std_mult', type=float, default=1,
                        help='Multiplier for std in lcb acquisition')

    parser.add_argument('--al_step_scale', type=str, default="log",
                        help='scale of spacing for active learning steps (log, linear)')
    parser.add_argument('--acquire_min', action='store_true',
                        help='if we should acquire min or max score molecules')
    parser.add_argument('--al_strategy', type=str, nargs='+',
                        choices=["random",
                                "explorative_greedy", "explorative_sample",
                                "score_greedy", "score_sample",
                                "exploit", "exploit_ucb", "exploit_lcb", "exploit_ts"],
                        default=["explorative_greedy"],
                        help='Strategy for active learning regime')
    parser.add_argument('--use_std', action='store_true', default=False,
                        help='Use std for evidence during active learning')

def update_checkpoint_args(args: Namespace):
    """
    Walks the checkpoint directory to find all checkpoints, updating args.checkpoint_paths and args.ensemble_size.

    :param args: Arguments.
    """
    if hasattr(args, 'checkpoint_paths') and args.checkpoint_paths is not None:
        return

    if args.checkpoint_dir is not None and args.checkpoint_path is not None:
        raise ValueError(
            'Only one of checkpoint_dir and checkpoint_path can be specified.')

    if args.checkpoint_dir is None:
        args.checkpoint_paths = [
            args.checkpoint_path] if args.checkpoint_path is not None else None
        return

    args.checkpoint_paths = []

    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname.endswith('.pt'):
                args.checkpoint_paths.append(os.path.join(root, fname))

    args.ensemble_size = len(args.checkpoint_paths)

    if args.ensemble_size == 0:
        raise ValueError(
            f'Failed to find any model checkpoints in directory "{args.checkpoint_dir}"')


def modify_predict_args(args: Namespace):
    """
    Modifies and validates predicting args in place.

    :param args: Arguments.
    """
    assert args.test_path
    assert args.preds_path
    assert args.checkpoint_dir is not None or args.checkpoint_path is not None or args.checkpoint_paths is not None

    update_checkpoint_args(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    # Create directory for preds path
    makedirs(args.preds_path, isfile=True)


def parse_predict_args() -> Namespace:
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args()
    modify_predict_args(args)

    return args


def modify_train_args(args: Namespace):
    """
    Modifies and validates training arguments in place.

    :param args: Arguments.
    """
    global temp_dir  # Prevents the temporary directory from being deleted upon function return

    # Load config file
    if args.config_path is not None:
        with open(args.config_path) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(args, key, value)

    assert args.data_path is not None
    assert args.dataset_type is not None

    if args.save_dir is not None:
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S%f")
        dataset = os.path.splitext(os.path.basename(args.data_path))[0]
        log_path = "{}_{}_{}".format(timestamp, dataset, args.confidence)
        args.save_dir = os.path.join(args.save_dir, log_path)

        if os.path.exists(args.save_dir):
            num_ctr = 0
            while (os.path.exists(f"{args.save_dir}_{num_ctr}")):
                num_ctr += 1
            args.save_dir = f"{args.save_dir}_{num_ctr}"

        makedirs(args.save_dir)
    else:
        temp_dir = TemporaryDirectory()
        args.save_dir = temp_dir.name

    if args.save_confidence is not None:
        args.save_confidence = os.path.join(args.save_dir, args.save_confidence)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    args.features_scaling = not args.no_features_scaling
    del args.no_features_scaling

    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        else:
            args.metric = 'rmse'

    if not ((args.dataset_type == 'classification' and args.metric in ['auc', 'prc-auc', 'accuracy']) or
            (args.dataset_type == 'regression' and args.metric in ['rmse', 'mae', 'r2'])):
        raise ValueError(
            f'Metric "{args.metric}" invalid for dataset type "{args.dataset_type}".')

    if (args.dataset_type=="regression" and args.confidence=="entropy"):
        raise ValueError(
            f"Confidence method {args.confidence} is not compatible with dataset type {args.dataset_type}")


    args.minimize_score = args.metric in ['rmse', 'mae']

    update_checkpoint_args(args)

    if args.features_only:
        assert args.features_generator or args.features_path

    args.use_input_features = args.features_generator or args.features_path

    if args.features_generator is not None and 'rdkit_2d_normalized' in args.features_generator:
        assert not args.features_scaling

    args.num_lrs = 1

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.hidden_size

    assert (args.split_type == 'predetermined') == (
        args.folds_file is not None) == (args.test_fold_index is not None)

    if args.test:
        args.epochs = 0


def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    temp_args, unk_args  = parser.parse_known_args()
    if temp_args.atomistic:
        add_atomistic_args(parser)
    args = parser.parse_args()
    modify_train_args(args)

    return args
