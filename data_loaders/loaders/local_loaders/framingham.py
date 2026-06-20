# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Framingham Heart Study — 10-year coronary heart disease (CHD) risk.
'''
from __future__ import annotations

from typing import Any

import pandas as pd
from data_loaders.utils import binarise_labels, impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


class FraminghamLoader(AbstractLoader):
    """Load the Framingham 10-year coronary heart disease dataset.

    Binary classification: predict whether a patient develops coronary heart
    disease within 10 years (class 1) or not (class 0) from demographic,
    behavioural and medical risk factors (age, smoking, blood pressure,
    cholesterol, BMI, glucose, diabetes, etc.). The classes are imbalanced
    (~15% positive) and several columns contain missing values which are
    median-imputed.

    Dataset stats: 4240 samples, 15 features.
    Source: Framingham Heart Study cohort (Kaggle "framingham.csv").

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
                         dataset_name='Framingham',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Load and preprocess the Framingham CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        df = pd.read_csv(self.local_dataset_path('framingham'))
        y = binarise_labels(df.pop('TenYearCHD'), {0: 0, 1: 1})
        return {
            'X': impute_missing(df),
            'y': y,
            'feature_names': df.columns.to_list(),
            'label_names': ['No CHD', 'CHD within 10yr'],
            'description': self.local_dataset_description('framingham'),
        }


if __name__ == "__main__":
    loader = FraminghamLoader()
    print(loader.get_info(long=True))
