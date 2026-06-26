# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
UCI Mammographic Mass dataset:
https://archive.ics.uci.edu/dataset/161/mammographic+mass
'''
from __future__ import annotations

from typing import Any

from data_loaders.utils import binarise_labels, impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict
from ucimlrepo import fetch_ucirepo


class MammographicMassLoader(AbstractLoader):
    """Load the UCI Mammographic Mass dataset.

    Binary classification: predict whether a mammographic mass is benign
    (class 0) or malignant (class 1) from the BI-RADS assessment, patient age
    and three mass attributes (shape, margin, density). The classes are roughly
    balanced (~46% malignant) but every feature column contains missing values,
    which are median-imputed during loading.

    Dataset stats: 961 samples, 5 features.
    Source: https://archive.ics.uci.edu/dataset/161/mammographic+mass

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
                         dataset_name='Mammographic Mass',
                         short_description='Mammography BI-RADS attributes and patient age for mass malignancy — binary',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Fetch and preprocess the Mammographic Mass dataset.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        mammographic = fetch_ucirepo(id=161)
        X = mammographic.data.features

        # raw 'Severity': 0 = benign, 1 = malignant
        y = binarise_labels(mammographic.data.targets['Severity'], {0: 0, 1: 1})
        return {
            'X': impute_missing(X),
            'y': y,
            'feature_names': X.columns.to_list(),
            'label_names': ['Benign', 'Malignant'],
            'description': str(mammographic.metadata),
        }


if __name__ == "__main__":
    loader = MammographicMassLoader()
    print(loader.get_info(long=True))
