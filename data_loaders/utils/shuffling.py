from __future__ import annotations

from typing import Any

import numpy as np


RANDOM_STATE = 42


def set_seed(seed: bool | int) -> None:
    """Set the NumPy random seed.

    Parameters
    ----------
    seed : bool or int
        If True, use the default random state (42). If False, use None
        (non-deterministic). If an int, use that value as the seed.
    """
    if seed == True:
        np.random.seed(seed=RANDOM_STATE)
    elif isinstance(seed, int):
        np.random.seed(seed=seed)
    elif seed == False:
        np.random.seed(seed=None)


def shuffle_data(data: dict[str, Any], seed: bool | int = True) -> dict[str, Any]:
    """Shuffle X and y arrays together in a data dict using sklearn.

    Parameters
    ----------
    data : dict
        Data dict with at least 'X' and 'y' numpy arrays.
    seed : bool or int, default=True
        Random seed. True uses the default state (42), False is
        non-deterministic, int uses that value.

    Returns
    -------
    dict
        Data dict with 'X' and 'y' shuffled in unison.
    """
    from sklearn.utils import shuffle

    if seed == True:
        seed = RANDOM_STATE
    data['X'], data['y'] = shuffle(
        data['X'], data['y'], random_state=seed)
    return data


def shuffle_dataset(data: dict[str, Any], seed: bool | int = True) -> dict[str, Any]:
    """Shuffle all numpy row arrays in a data dict in unison.

    All numpy arrays whose first dimension matches the number of instances
    are shuffled with the same permutation.

    Parameters
    ----------
    data : dict
        Data dict containing numpy arrays to shuffle (must include 'X').
    seed : bool or int, default=True
        Random seed passed to ``set_seed``.

    Returns
    -------
    dict
        Data dict with all matching numpy arrays shuffled in unison.
    """
    instances = data['X'].shape[0]
    # get random order
    set_seed(seed)
    p = np.random.permutation(instances)
    for key in data.keys():
        # apply to all numpy arrays that are data rows
        if type(data[key]) == np.ndarray and data[key].shape[0] == instances:
            # apply the shuffle
            data[key] = data[key][p]
    return data
