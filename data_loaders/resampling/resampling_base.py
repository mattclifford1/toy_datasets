from abc import ABC, abstractmethod

import numpy as np


class AbstractResampler(ABC):
    """Abstract base class for resampling strategies."""

    @abstractmethod
    def __call__(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample X and y to balance class counts.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Label array.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Resampled (X, y).
        """
        ...

    @abstractmethod
    def __repr__(self) -> str: ...