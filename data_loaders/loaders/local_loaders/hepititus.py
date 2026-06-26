# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for hepititus: https://archive.ics.uci.edu/dataset/46/hepatitis
'''
from __future__ import annotations

from typing import Any

import pandas as pd
from data_loaders.utils import binarise_labels
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


class HepatitisLoader(AbstractLoader):
    """Load the UCI Hepatitis dataset.

    Binary classification: predict whether a hepatitis patient Survived
    (class 0) or Died (class 1).  Columns with high missingness
    (PROTIME, ALKPHOSPHATE, ALBUMIN) are dropped, and rows containing
    missing-value markers (``'?'``) are removed.

    Dataset stats: ~137 samples (after filtering), ~16 features.
    Source: https://archive.ics.uci.edu/dataset/46/hepatitis

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                #  minority_reduce_scaler=5,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                        #  minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='Hepatitis',
                         short_description='Clinical features for hepatitis patient survival prediction — binary',
                         **kwargs)
        
    def load_data(self) -> DataDict:
        """Load and clean the Hepatitis CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        data = {}
        df = pd.read_csv(self.local_dataset_path('hepititus'))
        df.pop('PROTIME')
        df.pop('ALKPHOSPHATE')
        df.pop('ALBUMIN')
        # df.pop('LIVERBIG')
        # df.pop('LIVERFIRM')
        df = df[~df.isin(['?']).any(axis=1)]
        # Class: 1 -> Died (1), 2 -> Survived (0)
        data['y'] = binarise_labels(df.pop('Class'), {1: 1, 2: 0})
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        data['X'] = df.to_numpy().astype(float)
        data['feature_names'] = df.columns.to_list()
        data['label_names'] = ['Survived', 'Died']
        data['description'] = self.local_dataset_description('hepititus')
        return data
        

if __name__ == "__main__":
    loader = HepatitisLoader()
    print(loader.get_info(long=True))
    # loader.plot_dataset()
    # loader.plot_train_test_split()