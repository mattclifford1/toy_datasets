from __future__ import annotations

from typing import Any

import sklearn.datasets
from data_loaders.synthetic_generators import _generic_sklearn_loader
from data_loaders.abstract_loader import AbstractLoader, DataDict

class sklearn_normal_generator(AbstractLoader):
    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 num_samples: int = 200,
                 normal_features: int = 20,
                 **kwargs: Any) -> None:
        self.num_samples = num_samples
        self.normal_features = normal_features
         # work out the split size and ratio from the numbers
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Sklearn Synthetic Classification (Normal)',
                         **kwargs)

    def load_data(self) -> DataDict:
        '''
        ** read docs to add more params here
        https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification        returns:
            - data: dict containing 'X', 'y'
        '''
        data = _generic_sklearn_loader(load_func=sklearn.datasets.make_classification,
                                        samples=self.num_samples,
                                        test=False,
                                        n_features=self.normal_features)
        return data


if __name__ == "__main__":
    loader = sklearn_normal_generator()
    # loader.plot_dataset()
    loader.plot_train_test_split()
