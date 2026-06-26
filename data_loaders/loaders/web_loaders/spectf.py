# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
UCI SPECTF Heart dataset:
https://archive.ics.uci.edu/dataset/96/spectf+heart
'''
from __future__ import annotations

from typing import Any

from data_loaders.utils import binarise_labels, impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict
from ucimlrepo import fetch_ucirepo


class SPECTFHeartLoader(AbstractLoader):
    """Load the UCI SPECTF Heart dataset.

    Binary classification: diagnose cardiac Single Proton Emission Computed
    Tomography (SPECT) images as normal (class 0) or abnormal (class 1). Each
    patient is summarised by 44 continuous features (count-density ratios at
    rest and under stress for 22 regions of interest). The classes are
    imbalanced (~79% abnormal).

    Dataset stats: 267 samples, 44 features.
    Source: https://archive.ics.uci.edu/dataset/96/spectf+heart

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
                         dataset_name='SPECTF Heart',
                         short_description='SPECT cardiac imaging features for heart disease diagnosis — binary',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Fetch and preprocess the SPECTF Heart dataset.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        spectf = fetch_ucirepo(id=96)
        X = spectf.data.features

        # raw 'diagnosis': 0 = normal, 1 = abnormal
        y = binarise_labels(spectf.data.targets['diagnosis'], {0: 0, 1: 1})
        return {
            'X': impute_missing(X),
            'y': y,
            'feature_names': X.columns.to_list(),
            'label_names': ['Normal', 'Abnormal'],
            'description': str(spectf.metadata),
        }


if __name__ == "__main__":
    loader = SPECTFHeartLoader()
    print(loader.get_info(long=True))
