# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Ionosphere Dataset requires the prediction of structure in the atmosphere given radar returns targeting free electrons in the ionosphere.
UCI dataset: https://archive.ics.uci.edu/ml/datasets/Ionosphere
instances: 351
attributes: 34
'''
from __future__ import annotations

import os
from typing import Any

import pandas as pd
from data_loaders.abstract_loader import AbstractLoader, DataDict


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class ionosphere_loader(AbstractLoader):
    """Load the UCI Ionosphere radar dataset.

    Binary classification: distinguish 'good' radar returns (complex structure
    in the ionosphere, class 1) from 'bad' returns (class 0).  Features are
    17 complex-valued (real + imaginary) pulse pairs from a phased-array radar.

    Dataset stats: 351 samples, 34 features.
    Source: https://archive.ics.uci.edu/ml/datasets/Ionosphere

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
                         dataset_name='Ionosphere',
                         **kwargs)
        
    def load_data(self) -> DataDict:
        """Load and preprocess the Ionosphere CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                        'datasets', 'Ionosphere', 'data.csv'), header=None)
        df.drop(df[df[0] == 'I'].index, inplace=True)
        df[34] = df[34].map({'b': 0, 'g': 1}).astype('int64')
        data['y'] = df.pop(34).to_numpy()  # type: ignore
        data['X'] = df.to_numpy()
        data['feature_names'] = []
        for i in range(1, 18):
            data['feature_names'].append(f'Pulse {i} real')
            data['feature_names'].append(f'Pulse {i} imaginary')
        data['label_names'] = ['bad', 'good']
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'Ionosphere', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        return data


if __name__ == "__main__":
    loader = ionosphere_loader()
    print(loader.get_info(long=True))
    # loader.plot_dataset()
    # loader.plot_train_test_split()
