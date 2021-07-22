from typing import Any, List, Optional

import numpy as np


class StandardScaler:
    """A StandardScaler normalizes a dataset.

    When fit on a dataset, the StandardScaler learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the StandardScaler subtracts the means and divides by the standard deviations.

    If using an atomic model, then treat the mean and std as atomwise scaling
    parameters
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None,
                 replace_nan_token: Any = None, atomwise: bool = False,
                 no_scale : bool = False):
        """
        Initialize StandardScaler, optionally with means and standard deviations precomputed.

        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: The token to use in place of nans.
        :param atomwise: If true, use an atomwise scaler 
        :param no_scale: If true, don't scale, only shift
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token
        self.atomwise = atomwise
        self.no_scale = no_scale

    def fit(self, X: List[List[float]], atomlens : Optional[int] = None) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis.

        :param X: A list of lists of floats.
        :param atomlens: An optional integer list of atom lengths
        :return: The fitted StandardScaler.
        """
        X = np.array(X).astype(float)
        if atomlens is not None and self.atomwise:
            atomlens = np.array(atomlens).astype(int).reshape(-1,1)
            X = X / atomlens

        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        if self.no_scale: self.stds = np.ones(self.means.shape)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X: List[List[float]], atomlens : Optional[int] = None):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats.
        :param atomlens: An optional list of integer atom lengths
        :return: The transformed data.
        """
        X = np.array(X).astype(float)
        transform_factor = self.means

        # If atomistic setting, multiple by number of atoms 
        if atomlens is not None and self.atomwise: 
            atomlens = np.array(atomlens).astype(int).reshape(-1,1)
            transform_factor = transform_factor * atomlens

        transformed_with_nan = (X - transform_factor) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X: List[List[float]], 
                          atomlens : Optional[int] = None):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :param atomlens: An optional list of atomlens
        :return: The inverse transformed data.
        """
        X = np.array(X).astype(float)

        transform_factor = self.means
        # If atomistic setting, multiple by number of atoms 
        if atomlens is not None and self.atomwise: 
            atomlens = np.array(atomlens).astype(int).reshape(-1, 1)
            transform_factor = transform_factor * atomlens

        transformed_with_nan = X * self.stds + transform_factor
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none
