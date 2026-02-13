from __future__ import annotations

import numpy as np


class Normaliser:
    """MinMax scaler fitted on training data, scaling features to [-1, 1].

    Parameters
    ----------
    train_X : np.ndarray
        Training feature matrix used to fit the scaler.
    """

    def __init__(self, train_X: np.ndarray) -> None:
        from sklearn import preprocessing

        self.scaler = preprocessing.MinMaxScaler(
            feature_range=(-1,1)).fit(train_X)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Transform a feature matrix using the fitted scaler.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to transform.

        Returns
        -------
        np.ndarray
            Scaled feature matrix with values in [-1, 1].
        """
        return self.scaler.transform(X)

    def transform_instance(self, X: np.ndarray) -> np.ndarray:
        """Transform a single feature vector using the fitted scaler.

        Parameters
        ----------
        X : np.ndarray
            1D feature vector to transform.

        Returns
        -------
        np.ndarray
            Scaled 1D feature vector with values in [-1, 1].
        """
        return self.scaler.transform([X])[0]
