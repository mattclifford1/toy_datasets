from __future__ import annotations

from typing import Any

import numpy as np

from data_loaders.utils.shuffling import set_seed, RANDOM_STATE


def proportional_split(
        data: dict[str, Any],
        train_size: float = 0.8,
        seed: bool | int = True,
        minority_reduce_scaler: int | None = None,
        equal_test: bool = False,
        minority_reduce_scaler_test: int | None = None,
        majority_max: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    '''
    create a train, test split that preserves the class distributions
    Minority class is assumed to be class 1, majority class is assumed to be class 0
        data: data dict holder (not modified)
        train_size: size of the train set (0.5 means equal train, test size)
        minority_reduce_scaler: if not None, scale down the minority class by this factor
        equal_test: if True, balance test set classes first (reduces majority to match minority count)
        minority_reduce_scaler_test: if not None, scale down minority class in test set (applied after equal_test if both set)
        majority_max: if not None, cap the majority class of the TRAIN split at
            this many instances. Applied before minority_reduce_scaler, so the
            requested imbalance ratio is taken against the capped count. Use it
            to keep a very large dataset trainable while preserving its natural
            imbalance (e.g. MIMIC-IV has 1.6M majority rows, which is hopeless
            for a kernel SVM but fine at 50k). The test split is untouched.

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

    if not isinstance(majority_max, type(None)) and majority_max < 1:
        raise ValueError(
            f'majority_max needs to be at least 1 instead of :{majority_max}')

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
        if cls == 0:  # majority class
            if not isinstance(majority_max, type(None)) and majority_max != False:
                # cap the train side only; the rest stays available to test
                split_point = min(split_point, majority_max)
        if cls == 1: # minority class
            if not isinstance(minority_reduce_scaler, type(None)) and minority_reduce_scaler != False:
                split_point = max(int(len(train_inds[0])/minority_reduce_scaler), 1)
            # never ask for more minority train points than exist: doing so
            # leaves this class an empty test list, and np.concatenate of an
            # empty python list yields a float array which then fails as an index
            split_point = min(split_point, len(cls_inds))
        train_inds.append(list(cls_inds[0:split_point]))

        test_inds.append(list(cls_inds[split_point:]))

    # concat all the inds from each class
    # dtype is forced because np.concatenate over python lists returns float64
    # when any of them is empty, and a float array cannot index
    train_inds_arr = np.concatenate(train_inds).astype(int)
    test_inds_arr = np.concatenate(test_inds).astype(int)
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
        drop = []
        for cls in classes:
            inds = np.arange(len(test_split['y']))
            inds = inds[test_split['y'] == cls]
            drop.extend(inds[max_inst:].tolist())
        _drop_rows(test_split, drop)

    # Step 2: minority_reduce_scaler_test — reduce minority class further (applies after equal_test if both set)
    if not isinstance(minority_reduce_scaler_test, type(None)) and minority_reduce_scaler_test != False:
        minority_cls = 1
        inds = np.arange(len(test_split['y']))
        minority_inds = inds[test_split['y'] == minority_cls]
        n_keep = max(int(len(minority_inds) / minority_reduce_scaler_test), 1)
        _drop_rows(test_split, minority_inds[n_keep:])

    return train_split, test_split


def _drop_rows(split: dict[str, Any], inds_drop) -> None:
    '''
    delete rows from EVERY row-aligned array in the split, in place

    Dropping from 'X' and 'y' alone silently desynchronises any other
    per-instance array the loader supplied - 'cost_matrix' on the costcla
    datasets is the case that bites, since it is (n, 4) and would keep the
    dropped rows' costs against the wrong samples.
    '''
    inds_drop = np.asarray(inds_drop, dtype=int)
    if len(inds_drop) == 0:
        return
    n = len(split['y'])
    for key, val in split.items():
        if isinstance(val, np.ndarray) and val.shape[0] == n:
            split[key] = np.delete(val, inds_drop, axis=0)
