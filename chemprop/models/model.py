from argparse import Namespace
import logging
from typing import Optional

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
import torch
import torch.nn as nn
import numpy as np

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights

import schnetpack as spk
from schnetpack import AtomsData
from schnetpack.datasets import QM9

class EvaluationDropout(nn.Dropout):
    def __init__(self, *args, **kwargs):
        super(EvaluationDropout, self).__init__(*args, **kwargs)
        self.inference_mode = False

    def set_inference_mode(self, val : bool):
        self.inference_mode = val

    def forward(self, input):
        if self.inference_mode:
            return nn.functional.dropout(input, p = 0)
        else:
            return nn.functional.dropout(input, p = self.p)


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, confidence: bool = False,
                 conf_type: Optional[str] = None):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        :param confidence: Whether confidence values should be predicted.
        :param conf_type: Str definition of what type of confidence to use
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        # NOTE: Confidence flag is only set if the model must handle returning
        # confidence internally and for evidential learning.
        self.confidence = confidence
        self.conf_type  = conf_type

        if self.classification:
            if self.conf_type == 'evidence':
                self.final_activation = nn.Identity()
            else:
                self.final_activation = nn.Sigmoid()

        self.use_last_hidden = True

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)
        self.args = args

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        first_linear_dim = args.hidden_size
        if args.use_input_features:
            first_linear_dim += args.features_dim

        # When using dropout for confidence, use dropouts for evaluation in addition to training.
        if args.confidence == 'dropout':
            self.dropout = EvaluationDropout(args.dropout)
        else:
            self.dropout = nn.Dropout(args.dropout)

        activation = get_activation_function(args.activation)

        output_size = args.output_size

        if self.confidence: # if confidence should be learned
            if args.confidence == 'evidence':
                if self.classification: # dirichlet
                    # For each task, output both the positive and negative
                    # evidence for that task
                    output_size *= 2
                else: # normal inverse gamma
                    # For each task, output the parameters of the NIG
                    # distribution (gamma, lambda, alpha, beta)
                    output_size *= 4
            else: # gaussian MVE
                # For each task output the paramters of the Normal
                # distribution (mu, var)
                output_size *= 2

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                self.dropout,
                nn.Linear(first_linear_dim, output_size)
            ]
        else:
            ffn = [
                self.dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 3):
                ffn.extend([
                    activation,
                    self.dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])

            ffn.extend([
                activation,
                self.dropout,
                nn.Linear(args.ffn_hidden_size, args.last_hidden_size),
            ])

            ffn.extend([
                activation,
                self.dropout,
                nn.Linear(args.last_hidden_size, output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        ffn = self.ffn if self.use_last_hidden else nn.Sequential(
            *list(self.ffn.children())[:-1])
        output = ffn(self.encoder(*input))

        if self.confidence:
            if self.conf_type == "evidence":
                if self.classification:
                    # Convert the outputs into the parameters of a Dirichlet
                    # distribution (alpha).
                    output = nn.functional.softplus(output) + 1

                else:
                    min_val = 1e-6

                    min_val = 1e-6
                    # Split the outputs into the four distribution parameters
                    means, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1]//4, dim=1)
                    lambdas = torch.nn.Softplus()(loglambdas) + min_val
                    alphas = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
                    betas = torch.nn.Softplus()(logbetas) + min_val

                    # Return these parameters as the output of the model
                    output = torch.stack((means, lambdas, alphas, betas),
                                         dim = 2).view(output.size())
            else:
                even_indices = torch.tensor(range(0, list(output.size())[1], 2))
                odd_indices = torch.tensor(range(1, list(output.size())[1], 2))

                if self.args.cuda:
                    even_indices = even_indices.cuda()
                    odd_indices = odd_indices.cuda()

                predicted_means = torch.index_select(output, 1, even_indices)
                predicted_confidences = torch.index_select(output, 1, odd_indices)
                capped_confidences = nn.functional.softplus(predicted_confidences)

                output = torch.stack((predicted_means, capped_confidences), dim = 2).view(output.size())

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training and self.use_last_hidden:
            output = self.final_activation(output)

        return output

class AtomisticModel(nn.Module):

    def __init__(self, confidence : bool = False,
                 conf_type : Optional[str] = None):
        """ Requires initialization objects

        Wrapper class to use self.model but also give extra evidence columns

        :param classification: Whether the model is a classification model.
        :param confidence: Whether confidence values should be predicted.
        :param conf_type: Str definition of what type of confidence to use
        """
        super(AtomisticModel, self).__init__()

        self.classification = False
        self.confidence = confidence
        self.conf_type = conf_type
        self.representation = None
        self.outputs = []
        self.model = None

    def create_representation(self, args: Namespace):
        """
        Create representation layer

        :params args: Namespace
        """
        # Message passing for n_interaction layers
        # n_atom_basis is the n_in
        self.representation = spk.representation.SchNet(
            n_atom_basis=args.n_atom_basis, n_filters=args.n_filters,
            n_gaussians=args.n_gaussians, n_interactions=args.n_interactions,
            cutoff=args.cutoff, cutoff_network=spk.nn.cutoff.CosineCutoff,
        )
        self.args = args

    def create_output_layers(self, args : Namespace, train_dataset, scaler):
        """
        train_dataset used for computing standardization

        Means and stddevs used in standardization
        atomrefs is also an elementwise residual that should be learned

        :param args: Arguments
        :param train_dataset: AtomisticDataset used for normalization values
        :param scaler: Scaler used for normalization values
        """

        # Use this for evidential parameters
        output_size = args.output_size
        if self.confidence: # if confidence should be learned
            if args.confidence == 'evidence':
                if self.classification: # dirichlet
                    # For each task, output both the positive and negative
                    # evidence for that task
                    output_size *= 2
                else: # normal inverse gamma
                    # For each task, output the parameters of the NIG
                    # distribution (gamma, lambda, alpha, beta)
                    output_size *= 4
            else: # gaussian MVE
                # For each task output the paramters of the Normal
                # distribution (mu, var)
                output_size *= 2

        # Build an output predictor network
        # Hardcode QM9.U0, do not feed in atomrefs since these are accounted
        # for when the dataset is loaded
        self.output_U0 = spk.atomistic.Atomwise(n_in=args.n_atom_basis,
                                                n_out=1, property=QM9.U0)

        self.outputs.append(self.output_U0)
        # Outputs for all non-mean predictions (that don't need to be noramlized)
        self.has_aux = False
        if output_size > 1:
            self.aux_outputs = spk.atomistic.Atomwise(n_in=args.n_atom_basis,
                                                      n_out= output_size - 1,
                                                      property="bayesian_outputs")
            self.outputs.append(self.aux_outputs)
            self.has_aux = True
        self.args = args

    def create_joint_output(self, args:Namespace):
        """

        Crate full_model

        :param args: Namespace
        """
        # Model output
        # OUTPUT is n_out (int) – output dimension of target property (default: 1)
        self.model = spk.AtomisticModel(representation=self.representation,
                                        output_modules=self.outputs)

    def forward(self, batch, *inputs):
        """ Forward pass """
        # Hardcode QM9.U0
        # dictionary with energy_U0 as the  key ("property=QM9.U0")

        if self.args.cuda:
            batch = {k: v.cuda()  for k, v in batch.items()}

        output = self.model(batch)
        mean_outs = output[QM9.U0]
        output_list = [mean_outs]
        if self.has_aux:
            aux_outs= output["bayesian_outputs"]
            output_list.append(aux_outs)

        output = torch.cat(output_list, dim=1)

        if self.confidence:
            if self.conf_type == "evidence":
                if self.classification:
                    # Convert the outputs into the parameters of a Dirichlet
                    # distribution (alpha).
                    output = nn.functional.softplus(output) + 1

                else:
                    min_val = 1e-6

                    # Split the outputs into the four distribution parameters
                    means, loglambdas, logalphas, logbetas = torch.split(output,
                                                                         output.shape[1]//4,
                                                                         dim=1)
                    lambdas = torch.nn.Softplus()(loglambdas) + min_val
                    alphas = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
                    betas = torch.nn.Softplus()(logbetas) + min_val

                    # Return these parameters as the output of the model
                    output = torch.stack((means, lambdas, alphas, betas),
                                         dim = 2).view(output.size())
            else:
                even_indices = torch.tensor(range(0, list(output.size())[1], 2))
                odd_indices = torch.tensor(range(1, list(output.size())[1], 2))

                if self.args.cuda:
                    even_indices = even_indices.cuda()
                    odd_indices = odd_indices.cuda()

                predicted_means = torch.index_select(output, 1, even_indices)
                predicted_confidences = torch.index_select(output, 1, odd_indices)
                capped_confidences = nn.functional.softplus(predicted_confidences)

                output = torch.stack((predicted_means, capped_confidences), dim = 2).view(output.size())

        return output

def build_model(args: Namespace, train_data = None, scaler = None) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :param train_data: Used for initialization of atomistic network
    :param scaler: Used for initialization of atomistic network
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size

    if args.atomistic:
        if args.confidence == 'nn':
            model = AtomisticModel(confidence=True,
                                   conf_type="nn")
        elif args.confidence == "evidence":
            model = AtomisticModel(confidence=True,
                                   conf_type="evidence")
        else:
            model = AtomisticModel()

        # Build model
        model.create_output_layers(args, train_data, scaler=scaler)
        model.create_representation(args)
        model.create_joint_output(args)

    else:
        is_classifier = args.dataset_type == 'classification'
        if args.confidence == 'nn':
            model = MoleculeModel(classification=is_classifier, confidence=True,
                                  conf_type="nn")
        elif args.confidence == "evidence":
            model = MoleculeModel(classification=is_classifier, confidence=True,
                                  conf_type="evidence")
        else:
            model = MoleculeModel(classification=is_classifier)
        model.create_encoder(args)
        model.create_ffn(args)
        initialize_weights(model)

    return model
