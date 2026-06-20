# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
UCI Z-Alizadeh Sani coronary artery disease (CAD) dataset:
https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani
'''
from __future__ import annotations

from typing import Any

import pandas as pd
from data_loaders.utils import binarise_labels, encode_categoricals, impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


class ZAlizadehSaniLoader(AbstractLoader):
    """Load the UCI Z-Alizadeh Sani coronary artery disease (CAD) dataset.

    Binary classification: predict whether a patient has coronary artery disease
    (class 1) or a normal angiogram (class 0) from 55 features spanning
    demographics, symptoms, examination findings, ECG signs and laboratory blood
    tests — a rich mix of numeric and categorical variables. The classes are
    imbalanced (~71% CAD).

    Categorical columns (Yes/No flags, sex, and multi-category fields such as
    BBB and VHD) are integer-encoded.

    Dataset stats: 303 samples, 55 features.
    Source: https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani

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
                         dataset_name='Z-Alizadeh Sani CAD',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Load and preprocess the Z-Alizadeh Sani CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        df = pd.read_csv(self.local_dataset_path('zalizadeh_sani'))
        # raw 'Cath': Cad = coronary artery disease, Normal = healthy angiogram
        y = binarise_labels(df.pop('Cath'), {'Normal': 0, 'Cad': 1})
        X = encode_categoricals(df)
        return {
            'X': impute_missing(X),
            'y': y,
            'feature_names': X.columns.to_list(),
            'label_names': ['Normal', 'CAD'],
            'description': self.local_dataset_description('zalizadeh_sani'),
        }


if __name__ == "__main__":
    loader = ZAlizadehSaniLoader()
    print(loader.get_info(long=True))
