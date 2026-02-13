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
        **kwargs: Any,
) -> dict[str, Any]:
    '''
    sample from the a sklearn synthetic dataset
    returns:
        - data: dict containing 'X', 'y'
    '''
    seed = 42
    if test == True and load_func != sklearn.datasets.make_blobs:
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
