# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
UCI Breast Cancer Coimbra dataset:
https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra
'''
from __future__ import annotations

from typing import Any

from data_loaders.utils import binarise_labels, impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict
from ucimlrepo import fetch_ucirepo


class BreastCancerCoimbraLoader(AbstractLoader):
    """Load the UCI Breast Cancer Coimbra dataset.

    Binary classification: distinguish healthy controls (class 0) from breast
    cancer patients (class 1) using nine routine blood-analysis biomarkers
    (age, BMI, glucose, insulin, HOMA, leptin, adiponectin, resistin, MCP-1).
    A small, fully observed dataset that is nearly balanced (~55% patients), of
    interest because the features are inexpensive blood measurements rather than
    imaging or biopsy.

    Dataset stats: 116 samples, 9 features.
    Source: https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra

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
                         dataset_name='Breast Cancer Coimbra',
                         short_description='Anthropometric and blood biomarkers for breast cancer diagnosis — binary',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Fetch and preprocess the Breast Cancer Coimbra dataset.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        coimbra = fetch_ucirepo(id=451)
        X = coimbra.data.features

        # raw 'Classification': 1 = healthy control, 2 = patient
        y = binarise_labels(coimbra.data.targets['Classification'], {1: 0, 2: 1})
        return {
            'X': impute_missing(X),
            'y': y,
            'feature_names': X.columns.to_list(),
            'label_names': ['Healthy control', 'Patient'],
            'description': str(coimbra.metadata),
        }


if __name__ == "__main__":
    loader = BreastCancerCoimbraLoader()
    print(loader.get_info(long=True))
