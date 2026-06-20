# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
UCI Indian Liver Patient Dataset (ILPD):
https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
'''
from __future__ import annotations

from typing import Any

from data_loaders.utils import binarise_labels, impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict
from ucimlrepo import fetch_ucirepo


class IndianLiverLoader(AbstractLoader):
    """Load the UCI Indian Liver Patient Dataset (ILPD).

    Binary classification: predict whether a patient has liver disease
    (class 1) or not (class 0) from age, sex, and a panel of blood-test
    measurements. The classes are imbalanced (~71% liver patients). The
    categorical ``Gender`` column is encoded (Female=0, Male=1) and the few
    missing ``A/G Ratio`` values are median-imputed.

    Dataset stats: 583 samples, 10 features.
    Source: https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset

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
                         dataset_name='Indian Liver Patient',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Fetch and preprocess the Indian Liver Patient Dataset.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        ilpd = fetch_ucirepo(id=225)
        X = ilpd.data.features.copy()
        X['Gender'] = binarise_labels(X['Gender'], {'Female': 0, 'Male': 1})

        # raw 'Selector': 1 = liver patient, 2 = non liver patient
        y = binarise_labels(ilpd.data.targets['Selector'], {1: 1, 2: 0})
        return {
            'X': impute_missing(X),
            'y': y,
            'feature_names': X.columns.to_list(),
            'label_names': ['No liver disease', 'Liver disease'],
            'description': str(ilpd.metadata),
        }


if __name__ == "__main__":
    loader = IndianLiverLoader()
    print(loader.get_info(long=True))
