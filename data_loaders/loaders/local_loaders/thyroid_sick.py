# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
UCI Thyroid Disease (sick) dataset:
https://archive.ics.uci.edu/dataset/102/thyroid+disease
'''
from __future__ import annotations

from typing import Any

import pandas as pd
from data_loaders.utils import binarise_labels, encode_categoricals, impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


class ThyroidSickLoader(AbstractLoader):
    """Load the UCI Thyroid Disease (sick) dataset.

    Binary classification: predict whether a patient is sick (referred for
    thyroid dysfunction, class 1) or negative (class 0) from demographics,
    clinical t/f flags and thyroid hormone assays (TSH, T3, TT4, T4U, FTI). A
    classic, highly imbalanced anomaly-detection benchmark (~6% positive).

    The near-empty ``TBG`` assay column is dropped, the sentinel ``'?'`` is
    treated as missing and median-imputed, and categorical columns (sex, t/f
    flags, referral source) are integer-encoded.

    Dataset stats: 2800 samples, 28 features.
    Source: https://archive.ics.uci.edu/dataset/102/thyroid+disease

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
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Thyroid Sick',
                         short_description='Clinical and lab measurements for thyroid disorder detection — binary',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Load and preprocess the Thyroid (sick) CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        df = pd.read_csv(self.local_dataset_path('thyroid_sick'), na_values='?')
        df = df.drop(columns=['TBG'])  # near-empty assay column
        y = binarise_labels(df.pop('class'), {'negative': 0, 'sick': 1})
        X = encode_categoricals(df)
        return {
            'X': impute_missing(X),
            'y': y,
            'feature_names': X.columns.to_list(),
            'label_names': ['negative', 'sick'],
            'description': self.local_dataset_description('thyroid_sick'),
        }


if __name__ == "__main__":
    loader = ThyroidSickLoader()
    print(loader.get_info(long=True))
