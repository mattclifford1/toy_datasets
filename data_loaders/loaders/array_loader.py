# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Loader for data already in memory.
'''
from __future__ import annotations

from typing import Any

import numpy as np

from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


class ArrayLoader(AbstractLoader):
    """Give in-memory ``X`` and ``y`` the full loader interface (splits, scaling, noise, plots).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Integer class labels of shape ``(n_samples,)``.
    dataset_name : str, default='Array'
        Name used in plots and info output.
    label_names : list[str] or None, default=None
        Human-readable class names.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 dataset_name: str = 'Array',
                 label_names: list[str] | None = None,
                 **kwargs: Any) -> None:
        if len(X) != len(y):
            raise ValueError(f'X has {len(X)} rows but y has {len(y)}')
        self._X = np.asarray(X)
        self._y = np.asarray(y).astype(np.int64)
        self._label_names = label_names
        super().__init__(dataset_name=dataset_name, **kwargs)

    def load_data(self) -> DataDict:
        """Return copies of the arrays given at construction."""
        data = {'X': self._X.copy(), 'y': self._y.copy()}
        if self._label_names is not None:
            data['label_names'] = list(self._label_names)
        return data
