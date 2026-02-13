from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.utils
from data_loaders import utils
from data_loaders.utils import set_seed
from data_loaders.abstract_loader import AbstractLoader, DataDict


class XORGenerator(AbstractLoader):
    """Generate a synthetic XOR classification dataset.

    Class 0 and class 1 points are drawn from two pairs of Gaussian clusters
    arranged in the four XOR quadrants so that the classes are not linearly
    separable.

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    num_samples : int or list[int], default=200
        Total number of samples (split equally between classes) or a
        per-class list ``[n_class0, n_class1]``.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 num_samples: int | list[int] = 200,
                 **kwargs: Any) -> None:
        self.num_samples = num_samples
         # work out the split size and ratio from the numbers
        super().__init__(shuffle=shuffle,
                         dataset_name='XOR Synthetic',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Generate the XOR dataset.

        Returns
        -------
        DataDict
            Dict with keys ``'X'`` (shape ``(n_samples, 2)``) and ``'y'``.
        """
        data = self._get_XOR_single()
        return data


    def _get_XOR_single(self) -> DataDict:
        """Build the XOR layout from two pairs of Gaussian clusters.

        Returns
        -------
        DataDict
            Dict with ``'X'`` and ``'y'`` for the full XOR dataset.
        """
        if isinstance(self.num_samples, int):
            num_samples = [self.num_samples//2, self.num_samples//2]
        else:
            num_samples = self.num_samples
        mu = 5
        cov = [[1, 0], [0, 1]]
        covs = [cov, cov]
        top_data = self._get_two_normal_classes(means=[[-mu, -mu], [mu, -mu]],
                                        covs=covs,
                                        num_samples=[num_samples[0]//2, num_samples[1]//2])
        bot_data = self._get_two_normal_classes(means=[[mu, mu], [-mu, mu]],
                                        covs=covs,
                                        num_samples=[num_samples[0]//2, num_samples[1]//2])

        X = np.vstack([top_data['X'], bot_data['X']])
        y = np.hstack([top_data['y'], bot_data['y']])

        return {'X': X, 'y': y}


    def _get_two_normal_classes(
            self,
            means: list[list[float]] = [[0, 0], [10, 10]],
            covs: list[list[list[float]]] = [[[1, 0], [0, 1]],
                        [[1, 1], [1, 1]]],
            num_samples: list[int] = [3, 2],
            seed: bool | int | None = None,
    ) -> DataDict:
        """Sample two multivariate Gaussian classes.

        Parameters
        ----------
        means : list of list[float]
            Mean vector for each class ``[mean0, mean1]``.
        covs : list of list[list[float]]
            Covariance matrix for each class ``[cov0, cov1]``.
        num_samples : list[int]
            Number of samples per class ``[n0, n1]``.
        seed : bool, int, or None
            Random seed passed to :func:`set_seed`.

        Returns
        -------
        DataDict
            Dict with ``'X'`` and ``'y'`` arrays.
        """
        labels = [0, 1]
        X = []
        y = []
        for mean, cov, num_sample, label in zip(means, covs, num_samples, labels):
            set_seed(seed)
            X.append(np.random.multivariate_normal(mean, cov, size=num_sample))
            y.append(np.ones(num_sample)*label)
        X = np.vstack(X)
        y = np.hstack(y)
        # X, y = sklearn.utils.shuffle(X, y, random_state=seed)
        return {'X': X, 'y': y}


if __name__ == "__main__":
    loader = XORGenerator(
        num_samples=500,
        train_size=0.5,
        minority_reduce_scaler=10,
        equal_test=True,
        minority_reduce_scaler_test=10,
        )
    plot = False
    if plot:
        loader.plot_train_test_split(terminal_plot=True)
        loader.plot_dataset(terminal_plot=True)
    else:
        loader.get_train_test_split()  # print out the stats of the train/test split
