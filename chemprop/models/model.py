from argparse import Namespace
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn

from .mpn import MPN
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, siamese: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        :param siamese: Whether the model is a siamese model (i.e. two copies of the
        same model which determine the distance between inputs).
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        self.siamese = siamese

        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        first_linear_dim = args.hidden_size
        if args.use_input_features:
            first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        if self.siamese:
            ffn = ffn[:-1]  # Drop final output layer

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                batch_2: Union[List[str], BatchMolGraph] = None) -> torch.Tensor:
        """
        Runs the MoleculeModel on input.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :param batch_2: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        Only used for siamese network.
        :return: The output of the MoleculeModel, a 1D torch tensor of length batch_size.
        """
        output = self.ffn(self.encoder(batch, features_batch))

        if self.siamese:
            output_2 = self.ffn(self.encoder(batch_2))
            # Dot product between final representations
            output = torch.bmm(output.unsqueeze(1), output_2.unsqueeze(2)).squeeze(1)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)

        return output


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size

    model = MoleculeModel(
        classification=args.dataset_type == 'classification',
        siamese=args.siamese
    )
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
