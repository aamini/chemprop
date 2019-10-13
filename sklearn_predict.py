from argparse import ArgumentParser

from chemprop.sklearn_predict import predict_sklearn


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to CSV file containing testing data for which predictions will be made')
    parser.add_argument('--preds_path', type=str, required=True,
                        help='Path to CSV file where predictions will be saved')
    parser.add_argument('--dataset_type', type=str, required=True, choices=['classification', 'regression'],
                        help='Type of dataset')
    parser.add_argument('--model_type', type=str, choices=['random_forest', 'svm'], required=True,
                        help='scikit-learn model to use')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint (.pkl file)')
    parser.add_argument('--radius', type=int, default=2,
                        help='Morgan fingerprint radius')
    parser.add_argument('--num_bits', type=int, default=2048,
                        help='Number of bits in morgan fingerprint')
    args = parser.parse_args()

    predict_sklearn(args)
