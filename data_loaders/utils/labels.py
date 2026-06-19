'''
Helpers for converting raw dataset labels into integer class labels.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
from __future__ import annotations

from typing import Any

import numpy as np


def binarise_labels(y: Any, mapping: dict[Any, int]) -> np.ndarray:
    """Remap raw class labels to integer class labels using an explicit mapping.

    A single helper for every label format the loaders encounter: numeric or
    string labels held in a NumPy array, a pandas ``Series``, or a list.  The
    result is built into a *fresh* integer array rather than reassigned in
    place, so the following all work without temporary sentinel values:

    - **value remap**: ``{0: 0, 1: 1, 2: 0}`` (e.g. collapse class 2 into 0)
    - **many-to-one merge**: ``{1: 1, 2: 0, 3: 0}``
    - **swap**: ``{0: 1, 1: 0}`` (the in-place ``y[y==0]=1; y[y==1]=0`` form
      is buggy because the first assignment collides with the second)
    - **string keys**: ``{'B': 0, 'M': 1}`` or ``{'ckd': 1, 'notckd': 0}``

    Parameters
    ----------
    y : array-like
        Raw labels (ints, strings, etc.). Converted with ``np.asarray``; a
        pandas ``Series`` index is ignored and values are taken in order.
    mapping : dict
        Maps each raw label value to its integer class. Every value present in
        ``y`` must appear as a key (keys absent from ``y`` are allowed and
        simply have no effect).

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_samples,)`` with the mapped class labels.

    Raises
    ------
    ValueError
        If ``y`` contains a value that is not a key in ``mapping``.
    """
    y = np.asarray(y)
    out = np.empty(len(y), dtype=int)
    assigned = np.zeros(len(y), dtype=bool)
    for raw_value, class_label in mapping.items():
        mask = y == raw_value
        out[mask] = class_label
        assigned |= mask
    if not assigned.all():
        unmapped = np.unique(y[~assigned]).tolist()
        raise ValueError(
            f"binarise_labels: no mapping provided for label value(s) {unmapped}"
        )
    return out
