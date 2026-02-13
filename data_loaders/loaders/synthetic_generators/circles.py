from __future__ import annotations

from typing import Any

import sklearn.datasets
from data_loaders.loaders.synthetic_generators import _generic_sklearn_loader
from data_loaders.abstract_loader import AbstractLoader, DataDict

class CirclesGenerator(AbstractLoader):
    """Generate a synthetic concentric circles classification dataset.

    Wraps :func:`sklearn.datasets.make_circles` to produce two concentric
    rings that are not linearly separable.

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    num_samples : int, default=200
        Total number of samples to generate.
    circles_noise : float, default=0.2
        Standard deviation of Gaussian noise added to the samples.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 num_samples: int = 200,
                 circles_noise: float = 0.2,
                 **kwargs: Any) -> None:
        self.num_samples = num_samples
        self.circles_noise = circles_noise
         # work out the split size and ratio from the numbers
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Circles Synthetic',
                         **kwargs)

    def load_data(self) -> DataDict:
        '''
        sample from the circles data distribution
        returns:
            - data: dict containing 'X', 'y'
        '''
        data = _generic_sklearn_loader(load_func=sklearn.datasets.make_circles,
                                        samples=self.num_samples,
                                        test=False,
                                        noise=self.circles_noise,
                                        factor=0.8)
        return data


if __name__ == "__main__":
    loader = CirclesGenerator()
    # loader.plot_dataset()
    loader.plot_train_test_split()
