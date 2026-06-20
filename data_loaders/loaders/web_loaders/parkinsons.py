# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
UCI Parkinsons dataset: https://archive.ics.uci.edu/dataset/174/parkinsons
'''
from __future__ import annotations

from typing import Any

from ucimlrepo import fetch_ucirepo
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


class ParkinsonsLoader(AbstractLoader):
    """Load the UCI Parkinsons voice-measurement dataset.

    Binary classification: distinguish people with Parkinson's disease
    (class 1) from healthy controls (class 0) using 22 biomedical voice
    measurements. The classes are imbalanced (~75% Parkinson's).

    Dataset stats: 195 samples, 22 features.
    Source: https://archive.ics.uci.edu/dataset/174/parkinsons

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
                         dataset_name='Parkinsons',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Fetch the Parkinsons dataset from the UCI ML repository.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        parkinsons = fetch_ucirepo(id=174)
        X = parkinsons.data.features
        y = parkinsons.data.targets['status']
        return {
            'X': X.to_numpy(dtype=float),
            'y': y.to_numpy().astype(int),
            'feature_names': X.columns.to_list(),
            'label_names': ['Healthy', "Parkinson's"],
            'description': str(parkinsons.metadata),
        }


if __name__ == "__main__":
    loader = ParkinsonsLoader()
    print(loader.get_info(long=True))
