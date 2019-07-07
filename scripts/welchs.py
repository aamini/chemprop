from argparse import ArgumentParser
from typing import List

import numpy as np
from scipy import stats


def welchs(mean1: List[float],
           std1: List[float],
           nobs1: List[int],
           mean2: List[float],
           std2: List[float],
           nobs2: List[int]):
    # Expand one number of observations to all
    if len(nobs1) == 1:
        nobs1 = nobs1 * len(mean1)

    if len(nobs2) == 1:
        nobs2 = nobs2 * len(mean2)

    assert len(mean1) == len(std1) == len(nobs1) == len(mean2) == len(std2) == len(nobs2)

    # Compute Welch's t-test p-values for each dataset based on mean, standard deviation, and number of observations
    pvalues = [
        stats.ttest_ind_from_stats(mean1=m1, std1=s1, nobs1=n1, mean2=m2, std2=s2, nobs2=n2, equal_var=False).pvalue
        for m1, s1, n1, m2, s2, n2 in zip(mean1, std1, nobs1, mean2, std2, nobs2)
    ]

    # Print Welch's p-values
    print(' | '.join(f'{pvalue:.4e}' for pvalue in pvalues))

    # Chi-squared statistic
    chisquare = -2 * np.sum(np.log(pvalues))

    # Degrees of freedom
    df = 2 * len(pvalues)

    # Two-sided p-value for chi-squared (sf = survival function)
    pvalue = stats.distributions.chi2.sf(chisquare, df=df)

    # One-sided p-value for chi-squared
    pvalue = pvalue / 2

    # Print p-value
    print(f'p = {pvalue:.4e}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mean1', type=float, nargs='+', required=True,
                        help='Means of distributions of 1st model')
    parser.add_argument('--std1', type=float, nargs='+', required=True,
                        help='Standard deviations of distributions of 1st model')
    parser.add_argument('--nobs1', type=int, nargs='+', required=True,
                        help='Number of observations each mean/std is constructed from in 1st model')
    parser.add_argument('--mean2', type=float, nargs='+', required=True,
                        help='Means of distributions of 2nd model')
    parser.add_argument('--std2', type=float, nargs='+', required=True,
                        help='Standard deviations of distributions of 2nd model')
    parser.add_argument('--nobs2', type=int, nargs='+', required=True,
                        help='Number of observations each mean/std is constructed from in 2nd model')
    args = parser.parse_args()

    welchs(
        mean1=args.mean1,
        std1=args.std1,
        nobs1=args.nobs1,
        mean2=args.mean2,
        std2=args.std2,
        nobs2=args.nobs2
    )
