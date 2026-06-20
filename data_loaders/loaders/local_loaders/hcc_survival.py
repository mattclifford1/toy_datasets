# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
UCI Hepatocellular Carcinoma (HCC) survival dataset:
https://archive.ics.uci.edu/dataset/423/hcc+survival
'''
from __future__ import annotations

from typing import Any

import pandas as pd
from data_loaders.utils import binarise_labels, impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


class HCCSurvivalLoader(AbstractLoader):
    """Load the UCI Hepatocellular Carcinoma (HCC) survival dataset.

    Binary classification: predict 1-year survival of HCC patients — lives
    (class 0) or dies (class 1) — from 49 demographic, risk-factor, comorbidity
    and laboratory features selected per the EASL-EORTC guidelines. A
    heterogeneous, partially missing dataset (~10% of values missing, sentinel
    ``'?'``) with mild imbalance (~38% deaths). Missing values are
    median-imputed.

    Dataset stats: 165 samples, 49 features.
    Source: https://archive.ics.uci.edu/dataset/423/hcc+survival

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
                         dataset_name='HCC Survival',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Load and preprocess the HCC survival CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        df = pd.read_csv(self.local_dataset_path('hcc_survival'), na_values='?')
        # raw 'Class': 1 = lives, 0 = dies; map dies to the positive class
        y = binarise_labels(df.pop('Class'), {1: 0, 0: 1})
        return {
            'X': impute_missing(df),
            'y': y,
            'feature_names': df.columns.to_list(),
            'label_names': ['Lives', 'Dies'],
            'description': self.local_dataset_description('hcc_survival'),
        }


if __name__ == "__main__":
    loader = HCCSurvivalLoader()
    print(loader.get_info(long=True))
