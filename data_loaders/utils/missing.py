'''
Helper for turning a raw dataframe (with missing or sentinel values) into a
clean numeric feature matrix.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
from __future__ import annotations

import numpy as np
import pandas as pd


def impute_missing(df: pd.DataFrame, strategy: str = 'median') -> np.ndarray:
    """Coerce a dataframe to a float matrix and fill in missing values.

    Centralises the missing-value handling shared by loaders whose raw files
    use sentinels (e.g. ``'?'``) or have empty cells, so individual loaders do
    not each re-implement coercion and imputation. Non-numeric strings are
    turned into ``NaN`` first, then every ``NaN`` is replaced with its column's
    summary statistic. Columns that are entirely missing fall back to ``0``.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw feature columns (may contain strings, sentinels or ``NaN``).
    strategy : {'median', 'mean'}, default='median'
        Per-column statistic used to fill missing values.

    Returns
    -------
    np.ndarray
        Float feature matrix of shape ``(n_samples, n_features)`` with no
        missing values.
    """
    numeric = df.apply(pd.to_numeric, errors='coerce')
    fill = numeric.median() if strategy == 'median' else numeric.mean()
    numeric = numeric.fillna(fill).fillna(0.0)
    return numeric.to_numpy(dtype=float)
