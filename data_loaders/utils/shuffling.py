from __future__ import annotations

from typing import Any

import numpy as np


RANDOM_STATE = 42


def resolve_seed(seed: bool | int | None) -> int | None:
    """Map the package's seed convention onto a plain integer seed or None.

    The one place the convention is interpreted; every seed consumer calls this.

    Parameters
    ----------
    seed : bool, int or None
        True means the default random state (``RANDOM_STATE``, 42). False or
        None means non-deterministic. Any integer, including 0 and 1, is that
        seed.

    Returns
    -------
    int or None
        The seed to hand to numpy or scikit-learn.

    Notes
    -----
    ``bool`` is a subclass of ``int`` in Python, so ``1 == True`` and
    ``0 == False``. Testing the convention with ``==`` therefore turned seed 1
    into the default seed 42 and seed 0 into no seed at all. The booleans are
    matched by identity here, before the integer case.
    """
    if seed is True:
        return RANDOM_STATE
    if seed is False or seed is None:
        return None
    if isinstance(seed, (int, np.integer)):
        return int(seed)
    raise TypeError(f'seed must be a bool, an int or None, not {type(seed).__name__}')


def set_seed(seed: bool | int | None) -> None:
    """Set the NumPy global random seed.

    Parameters
    ----------
    seed : bool, int or None
        True is the default random state (42), an int is that seed, False
        reseeds from system entropy. None leaves the global stream exactly as
        it is - it always has, and callers rely on it to mean "do not touch".
        See :func:`resolve_seed` for the rest of the convention.
    """
    if seed is None:
        return
    np.random.seed(seed=resolve_seed(seed))


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

    # resolve_seed, not a bare `seed == True`: sklearn reads random_state=False
    # as the integer 0, so "non-deterministic" used to mean a fixed seed of 0
    data['X'], data['y'] = shuffle(
        data['X'], data['y'], random_state=resolve_seed(seed))
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
