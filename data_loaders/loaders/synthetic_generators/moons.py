from __future__ import annotations

from typing import Any

import sklearn.datasets
from data_loaders.loaders.synthetic_generators import _generic_sklearn_loader
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict

class MoonsGenerator(AbstractLoader):
    """Generate a synthetic two-moons (half-moons) classification dataset.

    Wraps :func:`sklearn.datasets.make_moons` to produce two interleaving
    half-circle clusters that are not linearly separable.

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    num_samples : int, default=200
        Total number of samples to generate.
    moons_noise : float, default=0.2
        Standard deviation of Gaussian noise added to the samples.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 num_samples: int = 200,
                 moons_noise: float = 0.2,
                 **kwargs: Any) -> None:
        self.num_samples = num_samples
        self.moons_noise = moons_noise
         # work out the split size and ratio from the numbers
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Moons Synthetic',
                         **kwargs)

    def load_data(self) -> DataDict:
        '''
        sample from the half moons data distribution
        returns:
            - data: dict containing 'X', 'y'
        '''
        data = _generic_sklearn_loader(load_func=sklearn.datasets.make_moons,
                                        samples=self.num_samples,
                                        test=False,
                                        noise=self.moons_noise)
        return data


if __name__ == "__main__":
    loader = MoonsGenerator()
    loader.plot_dataset(terminal_plot=True)
    loader.plot_train_test_split(terminal_plot=True)
