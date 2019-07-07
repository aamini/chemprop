from argparse import ArgumentParser
import csv

from scipy.stats import wilcoxon


def wilcoxon_significance(results_path: str, split_type: str, print_header: bool):
    split_type = split_type.lower()

    # Load results data
    with open(results_path) as f:
        data = list(csv.DictReader(f))

    experiments = [experiment for experiment in data[0].keys() if experiment != 'Fold']
    results = {experiment: [] for experiment in experiments}

    for row in data:
        # Skip incorrect split type
        if split_type not in row['Fold'].lower():
            continue

        for experiment, value in row.items():
            if experiment != 'Fold':
                try:
                    results[experiment].append(float(value))
                except ValueError:
                    pass

    if print_header:
        for i in range(len(experiments)):
            for j in range(i + 1, len(experiments)):
                print(f'{experiments[i]} vs {experiments[j]}', end='\t')
        print()

    for i in range(len(experiments)):
        for j in range(i + 1, len(experiments)):
            exp_1, exp_2 = results[experiments[i]], results[experiments[j]]

            if len(exp_1) == 0 or len(exp_2) == 0:
                print('', end='\t')
                continue

            print(wilcoxon(exp_1, exp_2).pvalue, end='\t')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to results .csv file in the structure of aggregate_results_by_dataset.py')
    parser.add_argument('--split_type', type=str, required=True,
                        help='Split type')
    parser.add_argument('--print_header', action='store_true', default=False,
                        help='Whether to print the header row of which experiments are being compared')
    args = parser.parse_args()

    wilcoxon_significance(
        results_path=args.results_path,
        split_type=args.split_type,
        print_header=args.print_header
    )
