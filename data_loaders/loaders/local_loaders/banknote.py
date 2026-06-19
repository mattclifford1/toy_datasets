# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
banknote authentication UCI dataset: https://archive.ics.uci.edu/ml/datasets/banknote+authentication#
instances: 1372
'''
from __future__ import annotations

from typing import Any

import pandas as pd
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


class BanknoteLoader(AbstractLoader):
    """Load the UCI Banknote Authentication dataset.

    Binary classification: distinguish authentic (class 0) from counterfeit
    (class 1) banknotes using four wavelet-transform features extracted from
    images.

    Dataset stats: 1372 samples, 4 features.
    Source: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.7
        Fraction of data used for training in train/test splits.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.7,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Banknote Authentication',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Load the Banknote Authentication CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        data = {}
        df = pd.read_csv(self.local_dataset_path('banknote_authentication'), header=None)
        data['y'] = df.pop(4).to_numpy()  # type: ignore
        data['X'] = df.to_numpy()
        data['feature_names'] = ['variance of Wavelet Transformed image',
                                'skewness of Wavelet Transformed image',
                                'curtosis of Wavelet Transformed image',
                                'entropy of image']
        data['label_names'] = ['Authentic', 'Counterfeit']
        data['description'] = self.local_dataset_description('banknote_authentication')
        return data
    
    
if __name__ == "__main__":
    loader = BanknoteLoader()
    print(loader.get_info(long=True))
    # loader.plot_dataset()
    # loader.plot_train_test_split()