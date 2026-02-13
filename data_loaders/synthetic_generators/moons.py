from __future__ import annotations

from typing import Any

import sklearn.datasets
from data_loaders.synthetic_generators import _generic_sklearn_loader
from data_loaders.abstract_loader import AbstractLoader, DataDict

class moons_generator(AbstractLoader):
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
    loader = moons_generator()
    loader.plot_dataset(terminal_plot=True)
    loader.plot_train_test_split(terminal_plot=True)
