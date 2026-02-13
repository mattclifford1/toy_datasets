from __future__ import annotations

from typing import Any

import sklearn.datasets
from data_loaders.loaders.synthetic_generators import _generic_sklearn_loader
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict

class BlobsGenerator(AbstractLoader):
    """Generate a synthetic isotropic Gaussian blobs classification dataset.

    Wraps :func:`sklearn.datasets.make_blobs` to produce well-separated
    Gaussian clusters (two classes by default).

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    num_samples : int, default=200
        Total number of samples to generate.
    blobs_features : int, default=2
        Number of feature dimensions for each sample.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 num_samples: int = 200,
                 blobs_features: int = 2,
                 **kwargs: Any) -> None:
        self.num_samples = num_samples
        self.blobs_features = blobs_features
         # work out the split size and ratio from the numbers
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Blobs Synthetic',
                         **kwargs)

    def load_data(self) -> DataDict:
        '''
        https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs
        returns:
            - data: dict containing 'X', 'y'
        '''
        data = _generic_sklearn_loader(load_func=sklearn.datasets.make_blobs,
                                        samples=self.num_samples,
                                        test=False,
                                        n_features=self.blobs_features)
        return data


if __name__ == "__main__":
    loader = BlobsGenerator()
    # loader.plot_dataset()
    loader.plot_train_test_split()
