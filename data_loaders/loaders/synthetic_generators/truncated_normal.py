from __future__ import annotations

from typing import Any

import numpy as np
import scipy
import sklearn.utils

from data_loaders.utils import resolve_seed
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


def truncated_normal(mean: float, std: float, bounds: list[float], num_samples: int) -> np.ndarray:
    # set up number of samples
    means = np.empty(num_samples)
    stds = np.empty(num_samples)
    boundss = np.empty([num_samples, 2])
    # apply deets
    means[:] = mean
    stds[:] = std
    boundss[:, 0] = bounds[0]
    boundss[:, 1] = bounds[1]
    # sample
    samples = scipy.stats.truncnorm.rvs((boundss[:, 0] - means) / stds, (boundss[:, 1] - means) / stds, loc=means, scale=stds)
    return np.expand_dims(samples, axis=1)

def multivar_truncated(
        means: list[float] = [0, 0],
        stds: list[float] = [1, 1],
        bounds: list[float] = [-10, 10],
        num_samples: int = 100,
) -> np.ndarray:
    # only does non correlated atm
    X = []
    for mean, std in zip(means, stds):
        X.append(truncated_normal(mean, std, bounds, num_samples))
    return np.hstack(X)

def get_two_classes(
        means: list[list[float]] = [[0,0], [10,10]],
        stds: list[list[float]] = [[1,1], [1,1]],
        bounds: list[list[float]] = [[-2,2], [8,12]],
        num_samples: list[int] = [3, 2],
) -> dict[str, Any]:
    labels = [0, 1]
    X = []
    y = []
    for mean, std, bound, num_sample, label in zip(means, stds, bounds, num_samples, labels):
        X.append(multivar_truncated(means=mean, stds=std, bounds=bound, num_samples=num_sample))
        y.append(np.ones(num_sample)*label)
    X = np.vstack(X)
    y = np.hstack(y)
    X, y = sklearn.utils.shuffle(X, y)  # , random_state=seed)
    return {'X': X, 'y':y}

def get_two_classes_R(
        means: list[list[float]] = [[0,0], [10,10]],
        stds: list[list[float]] = [[1,1], [1,1]],
        Rs: list[int] = [2, 2],
        num_samples: list[int] = [3, 2],
) -> None:
    bounds = 1 # write this
    get_two_classes(means=[[0,0], [10,10]], stds=[[1,1], [1,1]], bounds=[[-2,2], [8,12]], num_samples=[3, 2])


class TruncatedNormalGenerator(AbstractLoader):
    """Two classes of independent normals truncated to a known box per class.

    Parameters
    ----------
    num_samples : list of int, default=[1000, 1000]
        Points in class 0 and class 1.
    means : list of list of float, default=[[0, 0], [2, 2]]
        Per-class, per-feature means; the feature count is ``len(means[0])``.
    stds : list of list of float, default=[[1, 1], [1, 1]]
        Per-class, per-feature standard deviations before truncation.
    bounds : list of list of float, default=[[-2, 2], [0, 4]]
        Per-class ``[low, high]`` applied to every feature: the known support.
    **kwargs
        Forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 num_samples: list[int] = [1000, 1000],
                 means: list[list[float]] = [[0, 0], [2, 2]],
                 stds: list[list[float]] = [[1, 1], [1, 1]],
                 bounds: list[list[float]] = [[-2, 2], [0, 4]],
                 **kwargs: Any) -> None:
        self.num_samples = list(num_samples)
        self.means = [list(m) for m in means]
        self.stds = [list(sd) for sd in stds]
        self.bounds = [list(b) for b in bounds]
        super().__init__(shuffle=shuffle,
                         dataset_name='Truncated Normal Synthetic',
                         short_description='Two truncated normals with known supports',
                         **kwargs)

    def load_data(self) -> DataDict:
        rng = np.random.default_rng(resolve_seed(self.set_seed))
        X, y = [], []
        for label, (n, mean, std, (lo, hi)) in enumerate(
                zip(self.num_samples, self.means, self.stds, self.bounds)):
            mean, std = np.asarray(mean, float), np.asarray(std, float)
            X.append(scipy.stats.truncnorm.rvs(
                (lo - mean) / std, (hi - mean) / std, loc=mean, scale=std,
                size=(n, len(mean)), random_state=rng))
            y.append(np.full(n, label))
        return {'X': np.vstack(X), 'y': np.concatenate(y),
                'description': 'Two truncated normals', 'bounds': self.bounds}

if __name__ == '__main__':
    data = get_two_classes(num_samples=[50, 100])
    print(data)
