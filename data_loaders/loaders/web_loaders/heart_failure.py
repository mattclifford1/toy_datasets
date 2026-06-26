# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
UCI Heart Failure Clinical Records dataset:
https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
'''
from __future__ import annotations

from typing import Any

from data_loaders.utils import binarise_labels, impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict
from ucimlrepo import fetch_ucirepo


class HeartFailureLoader(AbstractLoader):
    """Load the UCI Heart Failure Clinical Records dataset.

    Binary classification: predict whether a heart-failure patient survived
    (class 0) or died (class 1) during the follow-up period, from clinical
    features (age, ejection fraction, serum creatinine, serum sodium, platelets,
    anaemia, diabetes, high blood pressure, smoking, sex, follow-up time). A
    small, clean and widely used benchmark with moderate imbalance (~32% deaths).

    Dataset stats: 299 samples, 12 features.
    Source: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records

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
                         dataset_name='Heart Failure',
                         short_description='Clinical records for heart failure event survival prediction — binary',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Fetch and preprocess the Heart Failure Clinical Records dataset.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        heart_failure = fetch_ucirepo(id=519)
        X = heart_failure.data.features

        # raw 'death_event': 0 = survived, 1 = died during follow-up
        y = binarise_labels(heart_failure.data.targets['death_event'], {0: 0, 1: 1})
        return {
            'X': impute_missing(X),
            'y': y,
            'feature_names': X.columns.to_list(),
            'label_names': ['Survived', 'Died'],
            'description': str(heart_failure.metadata),
        }


if __name__ == "__main__":
    loader = HeartFailureLoader()
    print(loader.get_info(long=True))
