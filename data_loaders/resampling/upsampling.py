from __future__ import annotations

from typing import Any

import numpy as np

from data_loaders.utils import set_seed
from data_loaders.resampling.resampling_base import AbstractResampler


class RandomDuplicateMinorityUpsampler(AbstractResampler):
    """Upsample minority classes by duplicating samples at random.

    Minority class is upsampled by a given factor or 'equal' to match majority class count.

    Parameters
    ----------
    random_state : int | bool, default=True
        Seed for reproducibility. Passed to :func:`set_seed`.
    sampling_strategy : str, default='auto'
        Only ``'auto'`` is supported: upsample all minority classes to the
        majority class count.
    """

    def __init__(
        self,
        random_state: int | bool = True,
        factor: float | str = 'equal',
    ) -> None:
        self.random_state = random_state
        self.factor = factor
        # check factor validity
        if isinstance(self.factor, (int, float)) and self.factor < 1:
            raise ValueError(f"Factor must be > 1 for upsampling. Got {self.factor}.")
        elif self.factor != 'equal' and not isinstance(self.factor, (int, float)):
            raise ValueError(f"Invalid factor: {self.factor}, either 'equal' or float > 1 expected.")

    def __repr__(self) -> str:
        return f'RandomDuplicate(factor={self.factor}, seed={self.random_state})'

    def __call__(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Duplicate minority-class rows until all classes are balanced.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Label array of shape (n_samples,).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Resampled (X, y) with equal class counts.
        """
        set_seed(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        if self.factor == 'equal':
            target_minority = int(counts.max())
        else:
            target_minority = int(counts.min() * self.factor)
        

        X_parts = [X]
        y_parts = [y]

        # upsample the minority classes by random duplication to the target count
        minority_cls = classes[counts == counts.min()][0]
        minority_count = counts.min()
        n_extra = target_minority - minority_count
        cls_inds = np.where(y == minority_cls)[0]
        extra_inds = np.random.choice(cls_inds, size=n_extra, replace=True)
        X_parts.append(X[extra_inds])
        y_parts.append(y[extra_inds])

        return np.concatenate(X_parts), np.concatenate(y_parts)


class SMOTEUpsampler(AbstractResampler):
    """Upsample minority classes using SMOTE (Synthetic Minority Over-sampling Technique).

    Requires ``imbalanced-learn`` (``pip install imbalanced-learn``).

    Parameters
    ----------
    random_state : int | bool, default=True
        Seed for reproducibility.
    k_neighbors : int, default=5
        Number of nearest neighbours for SMOTE interpolation.
    """

    def __init__(
        self,
        random_state: int | bool = True,
        k_neighbors: int = 5,
    ) -> None:
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def __repr__(self) -> str:
        return 'SMOTE'

    def __call__(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic minority-class samples via SMOTE.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Label array of shape (n_samples,).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Resampled (X, y) with balanced classes.

        Raises
        ------
        ImportError
            If ``imbalanced-learn`` is not installed.
        """
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise ImportError(
                "SMOTEUpsampler requires 'imbalanced-learn'. "
                "Install it with: pip install imbalanced-learn"
            ) from exc

        seed = self.random_state if isinstance(self.random_state, int) else None
        smote = SMOTE(random_state=seed, k_neighbors=self.k_neighbors)
        return smote.fit_resample(X, y)

