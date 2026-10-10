'''
Seeded corruptions of a dataset: irrelevant features and flipped labels.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
from __future__ import annotations

import numpy as np

from data_loaders.utils.shuffling import resolve_seed


def add_noise_features(X: np.ndarray, n_features: int, seed: bool | int | None = True) -> np.ndarray:
    """Append ``n_features`` columns of standard normal noise to ``X``.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_dims)``.
    n_features : int
        Number of N(0, 1) columns to append.
    seed : bool, int or None, default=True
        Seed convention of :func:`resolve_seed`.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, n_dims + n_features)``.
    """
    if n_features < 0:
        raise ValueError(f'noise_features must be non-negative, not {n_features}')
    noise = np.random.default_rng(resolve_seed(seed)).standard_normal((X.shape[0], n_features))
    return np.hstack([X, noise.astype(X.dtype, copy=False)])


def flip_labels(y: np.ndarray, fraction: float, seed: bool | int | None = True) -> np.ndarray:
    """Return a copy of ``y`` with a random ``fraction`` of rows moved to a different class.

    Parameters
    ----------
    y : np.ndarray
        Integer class labels.
    fraction : float
        Fraction of rows to relabel, in ``[0, 1]``.
    seed : bool, int or None, default=True
        Seed convention of :func:`resolve_seed`.

    Returns
    -------
    np.ndarray
        Labels with ``round(fraction * len(y))`` entries changed, each to a
        uniformly chosen other class present in ``y``.
    """
    if not 0 <= fraction <= 1:
        raise ValueError(f'label_noise must be in [0, 1], not {fraction}')
    y = np.asarray(y).copy()
    classes = np.unique(y)
    if len(classes) < 2:
        return y
    rng = np.random.default_rng(resolve_seed(seed))
    idx = rng.choice(len(y), int(round(fraction * len(y))), replace=False)
    # shift each picked label to one of the other classes, uniformly
    pos = np.searchsorted(classes, y[idx])
    y[idx] = classes[(pos + rng.integers(1, len(classes), len(idx))) % len(classes)]
    return y
