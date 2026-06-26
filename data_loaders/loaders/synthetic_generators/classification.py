from __future__ import annotations

from typing import Any

import sklearn.datasets
from data_loaders.loaders.synthetic_generators import _generic_sklearn_loader
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict

class SklearnNormalGenerator(AbstractLoader):
    """Generate a synthetic classification dataset via sklearn's make_classification.

    Wraps :func:`sklearn.datasets.make_classification` to produce a
    higher-dimensional normally-distributed binary classification dataset with
    redundant and informative features.

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    num_samples : int, default=200
        Total number of samples to generate.
    normal_features : int, default=20
        Total number of features (informative + redundant) per sample.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

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
                         short_description='Sklearn make_classification with Gaussian cluster features',
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
                                        n_features=self.normal_features,
                                        seed=self.set_seed,
                                        )
        return data


if __name__ == "__main__":
    loader = SklearnNormalGenerator()
    # loader.plot_dataset()
    loader.plot_train_test_split()
