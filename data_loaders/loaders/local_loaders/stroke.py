# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Stroke Prediction dataset (Kaggle, fedesoriano).
'''
from __future__ import annotations

from typing import Any

import pandas as pd
from data_loaders.utils import binarise_labels, encode_categoricals, impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


class StrokeLoader(AbstractLoader):
    """Load the Stroke Prediction dataset.

    Binary classification: predict whether a patient has had a stroke (class 1)
    or not (class 0) from demographic and clinical attributes (age, hypertension,
    heart disease, average glucose level, BMI, smoking status, work/residence
    type, etc.). Strongly imbalanced (~5% positive) with missing BMI values,
    making it a good stress test for class-balancing and missing-data handling.

    Categorical columns (gender, ever_married, work type, residence, smoking
    status) are integer-encoded and missing BMI values are median-imputed.

    Dataset stats: 5110 samples, 10 features.
    Source: Kaggle "Stroke Prediction Dataset" (fedesoriano).

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
                         dataset_name='Stroke',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Load and preprocess the Stroke CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        df = pd.read_csv(self.local_dataset_path('stroke'))
        y = binarise_labels(df.pop('stroke'), {0: 0, 1: 1})
        X = encode_categoricals(df)
        return {
            'X': impute_missing(X),
            'y': y,
            'feature_names': X.columns.to_list(),
            'label_names': ['No stroke', 'Stroke'],
            'description': self.local_dataset_description('stroke'),
        }


if __name__ == "__main__":
    loader = StrokeLoader()
    print(loader.get_info(long=True))
