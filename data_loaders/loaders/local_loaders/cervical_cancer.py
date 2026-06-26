# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
UCI Cervical Cancer (Risk Factors) dataset:
https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors
This dataset focuses on the prediction of indicators/diagnosis of cervical
cancer. The features cover demographic information, habits, and historic
medical records.
'''
from __future__ import annotations

from typing import Any

import pandas as pd
from data_loaders.utils import impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict

# screening-result columns; all but the chosen target are dropped as features
SCREENING_TARGETS = ['Hinselmann', 'Schiller', 'Citology', 'Biopsy']


class CervicalCancerLoader(AbstractLoader):
    """Load the UCI Cervical Cancer (Risk Factors) dataset.

    Binary classification: predict a positive cervical-cancer biopsy (class 1)
    from demographic and medical-history risk factors. Strongly imbalanced —
    only ~6% of patients have a positive biopsy, making it a good rare-disease
    example. Missing values (recorded as ``'?'``) are median-imputed and the
    other three screening-result columns are dropped.

    Dataset stats: 858 samples, 32 features.
    Source: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    target : str, default='Biopsy'
        Which screening result to use as the label (one of ``'Hinselmann'``,
        ``'Schiller'``, ``'Citology'``, ``'Biopsy'``).
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 target: str = 'Biopsy',
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Cervical Cancer',
                         short_description='Risk factors and test results for cervical cancer biopsy prediction — binary',
                         **kwargs)
        self.target = target

    def load_data(self) -> DataDict:
        """Load the bundled Cervical Cancer CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        df = pd.read_csv(self.local_dataset_path('cervical_cancer'), na_values='?')
        y = df.pop(self.target).to_numpy().astype(int)
        for col in SCREENING_TARGETS:
            if col in df.columns:
                df.pop(col)
        return {
            'X': impute_missing(df),
            'y': y,
            'feature_names': df.columns.to_list(),
            'label_names': ['Healthy', 'Cervical cancer'],
            'description': self.local_dataset_description('cervical_cancer'),
        }


if __name__ == "__main__":
    loader = CervicalCancerLoader()
    print(loader.get_info(long=True))
