from argparse import ArgumentParser, Namespace
import csv
import os

from scipy.stats import spearmanr


su3327_smiles = 'Nc1nnc(Sc2ncc(s2)[N+]([O-])=O)s1'


def compare_ranks(args: Namespace):
    with open(args.keep_path) as f:
        keep_smiles = set(line.strip() for line in f if line.strip() != '')

    with open(args.reference_path) as f:
        reference_data = list(csv.reader(f))[1:]

    with open(args.test_path) as f:
        test_data = list(csv.reader(f))[1:]

    reference_data = [row for row in reference_data if row[0] in keep_smiles]
    test_data = [row for row in test_data if row[0] in keep_smiles]

    reference_smiles, reference_preds = zip(*reference_data)
    test_smiles, test_preds = zip(*test_data)

    assert reference_smiles == test_smiles

    print(f'Rank correlation = {spearmanr(reference_preds, test_preds)}')

    reference_smiles_sorted = [smiles for smiles, pred in sorted(reference_data, key=lambda row: float(row[1]), reverse=True)]
    test_smiles_sorted = [smiles for smiles, pred in sorted(test_data, key=lambda row: float(row[1]), reverse=True)]

    print(f'Rank of su3327 in {os.path.basename(args.reference_path)} = {reference_smiles_sorted.index(su3327_smiles) + 1}')
    print(f'Rank of su3327 in {os.path.basename(args.test_path)} = {test_smiles_sorted.index(su3327_smiles) + 1}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--reference_path', type=str, required=True,
                        help='Path to .csv file containing smiles and prediction values of reference model')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to .csv file containing smiles and prediction values of test model')
    parser.add_argument('--keep_path', type=str, required=True,
                        help='Path to .txt file containing smiles that should be kept in the rankings')
    args = parser.parse_args()

    compare_ranks(args)
