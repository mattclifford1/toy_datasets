# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
UCI Arrhythmia dataset (fetched via OpenML):
https://archive.ics.uci.edu/dataset/5/arrhythmia
'''
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml
from data_loaders.utils import binarise_labels, impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict

# standard UCI arrhythmia class descriptions, keyed by the original 1-16 code
ARRHYTHMIA_CLASSES = {
    1: 'Normal',
    2: 'Ischemic changes (CAD)',
    3: 'Old anterior myocardial infarction',
    4: 'Old inferior myocardial infarction',
    5: 'Sinus tachycardy',
    6: 'Sinus bradycardy',
    7: 'Ventricular premature contraction',
    8: 'Supraventricular premature contraction',
    9: 'Left bundle branch block',
    10: 'Right bundle branch block',
    11: '1st degree AV block',
    12: '2nd degree AV block',
    13: '3rd degree AV block',
    14: 'Left ventricule hypertrophy',
    15: 'Atrial fibrillation or flutter',
    16: 'Others',
}


class ArrhythmiaLoader(AbstractLoader):
    """Load the UCI Arrhythmia ECG dataset.

    Multi-class classification of cardiac arrhythmia from 279 ECG-derived
    features. The 16 clinical categories are highly imbalanced, with several
    rare classes — useful for studying rare-class behaviour. Missing values are
    median-imputed.

    In the default binary mode normal recordings are class 0 and any arrhythmia
    is class 1. With ``binary=False`` the native classes present in the data are
    kept and remapped to contiguous integer labels.

    Dataset stats: 452 samples, 279 features.
    Source: https://archive.ics.uci.edu/dataset/5/arrhythmia

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    binary : bool, default=True
        If True, normal (class 0) vs. any arrhythmia (class 1).
        If False, keep the native multi-class labels.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    default_dim_reducer: str = 'TSNE'

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 binary: bool = True,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Arrhythmia',
                         short_description='ECG-derived features across cardiac arrhythmia classes (279 features)',
                         **kwargs)
        self.binary = binary

    def load_data(self) -> DataDict:
        """Fetch and preprocess the Arrhythmia dataset from OpenML.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'label_names'``, and
            ``'description'``.
        """
        arrhythmia = fetch_openml('arrhythmia', version=1, as_frame=True)
        X = impute_missing(arrhythmia.data)
        raw = arrhythmia.target.to_numpy().astype(int)

        if self.binary:
            # class 1 = normal -> 0, everything else -> arrhythmia (1)
            y = (raw != 1).astype(int)
            label_names = ['Normal', 'Arrhythmia']
        else:
            present = sorted(np.unique(raw).tolist())
            mapping = {code: i for i, code in enumerate(present)}
            y = binarise_labels(raw, mapping)
            label_names = [ARRHYTHMIA_CLASSES[code] for code in present]

        return {
            'X': X,
            'y': y,
            'label_names': label_names,
            'description': 'UCI Arrhythmia: 279 ECG-derived features across imbalanced cardiac arrhythmia classes.',
        }


if __name__ == "__main__":
    loader = ArrhythmiaLoader()
    print(loader.get_info(long=True))
