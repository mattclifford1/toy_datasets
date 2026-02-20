'''
Generate synthetic data from sklearn datasets
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sklearn.datasets
import sklearn.utils
from data_loaders import utils


def _generic_sklearn_loader(
        load_func: Callable[..., Any],
        samples: int = 200,
        test: bool = False,
        seed: bool | None | int = None,
        **kwargs: Any,
) -> dict[str, Any]:
    """Sample from a sklearn synthetic dataset generator.

    Parameters
    ----------
    load_func : callable
        A sklearn generator such as ``sklearn.datasets.make_moons``.
    samples : int, default=200
        Number of samples to generate.
    test : bool, default=False
        If True, offset the random seed by 1 so train and test sets differ.
    **kwargs
        Extra keyword arguments forwarded directly to ``load_func``.

    Returns
    -------
    dict
        Data dict with keys ``'X'`` and ``'y'``.
    """
    if seed == True:
        seed = 42
    elif seed == False:
        seed = None
    
    if isinstance(seed, int) and test == True and load_func != sklearn.datasets.make_blobs:
        seed += 1

    X, y = load_func(
        n_samples=samples,
        random_state=seed,
        shuffle=False,
        **kwargs
        )

    X, y = sklearn.utils.shuffle(X, y, random_state=seed)
    data = {'X': X, 'y':y}
    return data
