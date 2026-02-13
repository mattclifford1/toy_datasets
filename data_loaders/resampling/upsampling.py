from __future__ import annotations

from typing import Any

import numpy as np

from data_loaders.utils import set_seed
from data_loaders.resampling.resampling_base import AbstractResampler





class RandomDuplicateUpsampler(AbstractResampler):
    """Upsample minority classes by duplicating samples at random.

    All classes are upsampled to match the majority class count.

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
        sampling_strategy: str = 'auto',
    ) -> None:
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy

    def __repr__(self) -> str:
        return 'RandomDuplicate'

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
        target = int(counts.max())

        X_parts = [X]
        y_parts = [y]

        for cls, count in zip(classes, counts):
            n_extra = target - count
            if n_extra <= 0:
                continue
            cls_inds = np.where(y == cls)[0]
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


def proportional_upsample(
    data: dict[str, Any],
    strategy: str = 'random',
    seed: bool | int = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Upsample data so all classes have equal representation.

    Mirrors the API of :func:`proportional_downsample`. All numpy arrays in
    *data* whose first dimension matches ``X``'s row count are resampled with
    the same indices.

    Parameters
    ----------
    data : dict[str, Any]
        Data dict with at least ``'X'`` and ``'y'`` keys.
    strategy : str, default='random'
        Upsampling strategy: ``'random'`` (duplicate rows) or ``'smote'``
        (synthetic samples).
    seed : bool | int, default=True
        Random seed passed to the chosen upsampler.
    **kwargs : Any
        Extra keyword arguments forwarded to the upsampler constructor.

    Returns
    -------
    dict[str, Any]
        Data dict with balanced ``'X'``, ``'y'``, and any co-indexed arrays.

    Raises
    ------
    ValueError
        If *strategy* is not ``'random'`` or ``'smote'``.
    """
    if strategy == 'random':
        upsampler: AbstractResampler = RandomDuplicateUpsampler(
            random_state=seed, **kwargs
        )
    elif strategy == 'smote':
        upsampler = SMOTEUpsampler(random_state=seed, **kwargs)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose 'random' or 'smote'."
        )

    X_orig = data['X']
    y_orig = data['y']
    n_orig = X_orig.shape[0]

    X_new, y_new = upsampler(X_orig, y_orig)

    # Build index map: for each new row, which original row does it come from?
    # Original rows are kept as-is (indices 0..n_orig-1 map to themselves).
    # Extra rows appended per class map back to the source row in X_orig.
    # We recover this by comparing X_new rows to X_orig rows.
    # Simpler: replay the same selection and track indices explicitly.
    # Re-derive the extra indices deterministically for consistent co-indexing.
    set_seed(seed)
    classes, counts = np.unique(y_orig, return_counts=True)
    target = int(counts.max())

    extra_src_inds: list[np.ndarray] = []
    for cls, count in zip(classes, counts):
        n_extra = target - count
        if n_extra <= 0:
            continue
        cls_inds = np.where(y_orig == cls)[0]
        extra_inds = np.random.choice(cls_inds, size=n_extra, replace=True)
        extra_src_inds.append(extra_inds)

    all_src_inds = np.concatenate(
        [np.arange(n_orig)]
        + (extra_src_inds if extra_src_inds else [np.array([], dtype=int)])
    )

    new_data = dict(data)
    new_data['X'] = X_new
    new_data['y'] = y_new

    for key, val in data.items():
        if key in ('X', 'y'):
            continue
        if isinstance(val, np.ndarray) and val.shape[0] == n_orig:
            new_data[key] = val[all_src_inds]

    return new_data
