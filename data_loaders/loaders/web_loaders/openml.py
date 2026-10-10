# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Binary tabular benchmarks fetched from OpenML: https://www.openml.org
'''
from __future__ import annotations

import os
from typing import Any

import numpy as np

from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict

# name -> OpenML data id, raw label of class 1 (the minority, or either if balanced),
# label names (class 0, class 1), one-line description, and any columns to drop.
# All numeric, no missing values.
OPENML_DATASETS: dict[str, dict[str, Any]] = {
    'Spambase': dict(data_id=44, positive='1', labels=('Ham', 'Spam'),
                     short='Word and character frequencies of emails — spam vs ham'),
    'Phoneme': dict(data_id=1489, positive='2', labels=('Nasal', 'Oral'),
                    short='Five harmonic amplitudes of spoken vowels — nasal vs oral'),
    'MAGIC Gamma Telescope': dict(data_id=1120, positive='h', labels=('Gamma', 'Hadron'),
                                  short='Cherenkov telescope image moments — gamma signal vs hadron background'),
    'EEG Eye State': dict(data_id=1471, positive='2', labels=('Eyes open', 'Eyes closed'),
                          short='14-channel EEG readings — eyes open vs closed'),
    'Default of Credit Card Clients': dict(data_id=42477, positive='1', labels=('No default', 'Default'),
                                           short='Taiwanese credit card payment history — default next month'),
    'QSAR Biodegradation': dict(data_id=1494, positive='2', labels=('Not biodegradable', 'Ready biodegradable'),
                                short='Molecular descriptors of chemicals — ready biodegradability'),
    'Bioresponse': dict(data_id=4134, positive='0', labels=('Response', 'No response'),
                        short='1776 molecular descriptors — biological response, high-dimensional'),
    'Hill-Valley': dict(data_id=1479, positive='1', labels=('Valley', 'Hill'),
                        short='100-point noiseless series — hill vs valley shape'),
    # V28-V33 one-hot the other six fault types, so they leak the label.
    'Steel Plates Fault': dict(data_id=1504, positive='2', labels=('Named fault', 'Other fault'),
                               short='Geometric and luminosity features of steel plate faults — other vs named type',
                               drop=['V28', 'V29', 'V30', 'V31', 'V32', 'V33']),
    'KC1 Software Defects': dict(data_id=1067, positive='true', labels=('No defect', 'Defect'),
                                 short='McCabe and Halstead code metrics of NASA modules — defect prediction'),
    'Blood Transfusion': dict(data_id=1464, positive='2', labels=('No donation', 'Donated'),
                              short='Donor recency, frequency and volume — donated in March 2007'),
    'Ozone Level 8hr': dict(data_id=1487, positive='2', labels=('Normal day', 'Ozone day'),
                            short='Weather measurements — 8-hour ozone alert days, highly imbalanced'),
}


class OpenMLLoader(AbstractLoader):
    """Load a binary numeric dataset from OpenML by its name in :data:`OPENML_DATASETS`.

    Downloads once to ``data_loaders/loaders/datasets/OpenML/`` and reads that
    cache afterwards. Class 1 is the listed positive label; every other label is
    class 0.

    Parameters
    ----------
    name : str
        Key of :data:`OPENML_DATASETS`, e.g. ``'Spambase'``.
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 name: str,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 **kwargs: Any) -> None:
        if name not in OPENML_DATASETS:
            raise ValueError(f'Unknown OpenML dataset {name!r}. Choose from: {list(OPENML_DATASETS)}')
        self.spec = OPENML_DATASETS[name]
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name=name,
                         short_description=self.spec['short'],
                         **kwargs)

    def load_data(self) -> DataDict:
        """Fetch (or read the cached copy of) the dataset.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        from sklearn.datasets import fetch_openml

        this_dir = os.path.dirname(os.path.abspath(__file__))
        bunch = fetch_openml(data_id=self.spec['data_id'], as_frame=True, parser='auto',
                             data_home=os.path.join(this_dir, '..', 'datasets', 'OpenML'))
        X = bunch.data.drop(columns=self.spec.get('drop', []))
        y = (bunch.target.astype(str).to_numpy() == self.spec['positive']).astype(np.int64)
        return {
            'X': X.to_numpy(dtype=np.float64),
            'y': y,
            'feature_names': X.columns.to_list(),
            'label_names': list(self.spec['labels']),
            'description': bunch.DESCR,
        }


if __name__ == '__main__':
    loader = OpenMLLoader('Spambase')
    print(loader.get_info(long=False))
