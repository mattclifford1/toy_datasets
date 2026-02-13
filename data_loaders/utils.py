from __future__ import annotations

from typing import Any

import numpy as np


RANDOM_STATE = 42


class normaliser:
    """MinMax scaler fitted on training data, scaling features to [-1, 1].

    Parameters
    ----------
    train_X : np.ndarray
        Training feature matrix used to fit the scaler.
    """

    def __init__(self, train_X: np.ndarray) -> None:
        from sklearn import preprocessing

        self.scaler = preprocessing.MinMaxScaler(
            feature_range=(-1,1)).fit(train_X)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Transform a feature matrix using the fitted scaler.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to transform.

        Returns
        -------
        np.ndarray
            Scaled feature matrix with values in [-1, 1].
        """
        return self.scaler.transform(X)

    def transform_instance(self, X: np.ndarray) -> np.ndarray:
        """Transform a single feature vector using the fitted scaler.

        Parameters
        ----------
        X : np.ndarray
            1D feature vector to transform.

        Returns
        -------
        np.ndarray
            Scaled 1D feature vector with values in [-1, 1].
        """
        return self.scaler.transform([X])[0]


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


def proportional_split(
        data: dict[str, Any],
        train_size: float = 0.8,
        seed: bool | int = True,
        minority_reduce_scaler: int | None = None,
        equal_test: bool = False,
        minority_reduce_scaler_test: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    '''
    create a train, test split that preserves the class distributions
    Minority class is assumed to be class 1, majority class is assumed to be class 0
        data: data dict holder (not modified)
        train_size: size of the train set (0.5 means equal train, test size)
        minority_reduce_scaler: if not None, scale down the minority class by this factor
        equal_test: if True, balance test set classes first (reduces majority to match minority count)
        minority_reduce_scaler_test: if not None, scale down minority class in test set (applied after equal_test if both set)

    returns: train_split, test_split (two new dicts)
    '''
    if train_size <= 0 or train_size > 1:
        raise ValueError(
            f'train_size needs to be between 0 and 1 instead of :{train_size}')
    if not isinstance(minority_reduce_scaler, type(None)) and minority_reduce_scaler < 1:
        raise ValueError(
            f'minority_reduce_scaler needs to be greater than 1 instead of :{minority_reduce_scaler}')
    if not isinstance(minority_reduce_scaler_test, type(None)) and minority_reduce_scaler_test < 1:
        raise ValueError(
            f'minority_reduce_scaler_test needs to be greater than 1 instead of :{minority_reduce_scaler_test}')

    set_seed(seed)
    # get current class proportions
    classes, counts = np.unique(data['y'], return_counts=True)
    classes = sorted(classes)
    test_inds: list[list[int]] = []
    train_inds: list[list[int]] = []
    for i, cls in enumerate(classes):
        # get all the inds of current class
        cls_inds = np.where(data['y'] == cls)[0]
        # shuffle all the inds to get a random selection
        set_seed(seed)
        np.random.shuffle(cls_inds)
        set_seed(seed)
        # now split the data inds into train/test
        split_point = int(counts[i]*train_size)
        if cls == 1: # minority class
            if not isinstance(minority_reduce_scaler, type(None)) and minority_reduce_scaler != False:
                split_point = max(int(len(train_inds[0])/minority_reduce_scaler), 1)
        train_inds.append(list(cls_inds[0:split_point]))

        test_inds.append(list(cls_inds[split_point:]))

    # concat all the inds from each class
    train_inds_arr = np.concatenate(train_inds)
    test_inds_arr = np.concatenate(test_inds)
    # now apply the split to all data arrays
    train_split: dict[str, Any] = {}
    test_split: dict[str, Any] = {}
    instances = data['X'].shape[0]
    for key, val in data.items():
        # apply to all numpy arrays that are data rows
        if isinstance(val, np.ndarray) and data[key].shape[0] == instances:
            # extract and store the splits
            train_split[key] = val[train_inds_arr]
            test_split[key] = val[test_inds_arr]
    # Step 1: equal_test — balance test classes first
    if equal_test == True:
        classes, counts = np.unique(test_split['y'], return_counts=True)
        max_inst = min(counts)
        for cls in classes:
            inds = np.arange(len(test_split['y']))
            inds = inds[test_split['y'] == cls]
            inds_drop = inds[max_inst:]
            test_split['y'] = np.delete(test_split['y'], inds_drop)
            test_split['X'] = np.delete(test_split['X'], inds_drop, axis=0)

    # Step 2: minority_reduce_scaler_test — reduce minority class further (applies after equal_test if both set)
    if not isinstance(minority_reduce_scaler_test, type(None)) and minority_reduce_scaler_test != False:
        minority_cls = 1
        inds = np.arange(len(test_split['y']))
        minority_inds = inds[test_split['y'] == minority_cls]
        n_keep = max(int(len(minority_inds) / minority_reduce_scaler_test), 1)
        inds_drop = minority_inds[n_keep:]
        test_split['y'] = np.delete(test_split['y'], inds_drop)
        test_split['X'] = np.delete(test_split['X'], inds_drop, axis=0)

    return train_split, test_split
