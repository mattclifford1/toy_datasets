'''
MADELON-style data from sklearn's make_classification, which follows
I. Guyon, "Design of experiments for the NIPS 2003 variable selection benchmark", 2003.
'''
from __future__ import annotations

from typing import Any

from sklearn.datasets import make_classification

from data_loaders import utils
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


class MadelonGenerator(AbstractLoader):
    """Guyon's MADELON design: clusters on the vertices of a 5-D hypercube, plus
    redundant and distractor features.

    Parameters
    ----------
    num_samples : int or list of int, default=[2000, 2000]
        Total points, or points in class 0 and class 1.
    n_features : int, default=500
        All features: 5 informative, 15 redundant, the rest noise.
    flip_y : float, default=0.01
        Fraction of labels flipped at random.
    **kwargs
        Forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 num_samples: int | list[int] = [2000, 2000],
                 n_features: int = 500,
                 flip_y: float = 0.01,
                 **kwargs: Any) -> None:
        if isinstance(num_samples, int):
            num_samples = [num_samples // 2, num_samples - num_samples // 2]
        self.num_samples = list(num_samples)
        self.n_features = n_features
        self.flip_y = flip_y
        super().__init__(shuffle=shuffle,
                         dataset_name='Madelon Synthetic',
                         short_description='Hypercube clusters with 480 distractor features',
                         **kwargs)

    def load_data(self) -> DataDict:
        n = sum(self.num_samples)
        X, y = make_classification(
            n_samples=n, n_features=self.n_features, n_informative=5,
            n_redundant=15, n_clusters_per_class=16, flip_y=self.flip_y,
            weights=[self.num_samples[0] / n], shuffle=False,
            random_state=utils.resolve_seed(self.set_seed))
        return {'X': X, 'y': y, 'description': 'MADELON (Guyon 2003)'}
