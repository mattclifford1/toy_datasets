# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
UCI Breast Cancer Wisconsin (Prognostic) dataset (WPBC):
https://archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic
'''
from __future__ import annotations

from typing import Any

from data_loaders.utils import binarise_labels, impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict
from ucimlrepo import fetch_ucirepo


class BreastCancerPrognosticLoader(AbstractLoader):
    """Load the UCI Breast Cancer Wisconsin (Prognostic) dataset (WPBC).

    Binary classification: predict whether breast cancer recurred (class 1) or
    remained non-recurrent (class 0) within the follow-up period. Features are
    the follow-up time plus 30 nuclear morphology measurements (mean, standard
    error and worst of radius, texture, perimeter, etc.) and two tumour
    descriptors (tumour size, lymph node status). Complements the *diagnostic*
    Wisconsin dataset; the classes are imbalanced (~24% recurrence) and the
    lymph node status has a few missing values which are median-imputed.

    Dataset stats: 198 samples, 33 features.
    Source: https://archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic

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
                         dataset_name='Breast Cancer Prognostic',
                         short_description='FNA features for breast cancer recurrence prediction — binary',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Fetch and preprocess the WPBC dataset.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        wpbc = fetch_ucirepo(id=16)
        X = wpbc.data.features

        # raw 'Outcome': N = non-recurrent, R = recurrent
        y = binarise_labels(wpbc.data.targets['Outcome'], {'N': 0, 'R': 1})
        return {
            'X': impute_missing(X),
            'y': y,
            'feature_names': X.columns.to_list(),
            'label_names': ['Non-recurrent', 'Recurrent'],
            'description': str(wpbc.metadata),
        }


if __name__ == "__main__":
    loader = BreastCancerPrognosticLoader()
    print(loader.get_info(long=True))
