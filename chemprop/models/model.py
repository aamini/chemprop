from argparse import Namespace

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
import torch.nn as nn

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights

class MoleculeModelEnsemble(nn.Module):
    def __init__(self, ensemble_size: int, classification: bool, gaussian: bool):
        super(MoleculeModelEnsemble, self).__init__()

        self.models = [MoleculeModel(classification) for _ in range(ensemble_size)]
        self.gaussian = GaussianProcessClassifier() if classification else GaussianProcessRegressor()

class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        self.use_last_layer = True

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

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        ffn = self.ffn if self.use_last_layer else list(self.ffn.children()[:-1])
        output = ffn(self.encoder(*input))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training and self.use_last_layer:
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

    model = MoleculeModel(classification=args.dataset_type == 'classification')
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
