from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AbstractDimReducer(ABC):
    """Abstract base class for all dimensionality reducers."""

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data into the reduced space.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to transform.

        Returns
        -------
        np.ndarray
            Transformed feature matrix.
        """
        ...
