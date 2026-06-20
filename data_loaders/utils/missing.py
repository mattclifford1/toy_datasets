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


def encode_categoricals(
        df: pd.DataFrame,
        exclude: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Integer-encode non-numeric columns of a dataframe.

    Centralises categorical encoding shared by loaders whose raw files mix
    numeric and string columns (e.g. ``'Yes'``/``'No'`` flags or multi-category
    fields), so individual loaders do not each re-implement it. Numeric columns
    are left untouched, as are numeric columns that merely carry a missing-value
    sentinel (e.g. a ``'?'`` among the numbers) so their continuous values
    survive for a later :func:`impute_missing` pass. Only genuinely categorical
    columns (no numeric entries at all) are encoded, with their categories mapped
    to integers in sorted order (so binary ``'N'``/``'Y'`` becomes ``0``/``1``
    deterministically). Missing values are preserved as ``NaN``.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw feature columns (mix of numeric and string columns).
    exclude : tuple of str, default=()
        Column names to leave unchanged (e.g. a label column).

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with categorical columns replaced by integer codes.
    """
    df = df.copy()
    for col in df.columns:
        if col in exclude or pd.api.types.is_numeric_dtype(df[col]):
            continue
        # leave numeric columns that only carry a sentinel (e.g. '?') for the
        # imputation pass; encode only columns with no numeric values at all.
        if pd.to_numeric(df[col], errors='coerce').notna().any():
            continue
        categories = sorted(df[col].dropna().unique())
        mapping = {category: code for code, category in enumerate(categories)}
        df[col] = df[col].map(mapping)
    return df
