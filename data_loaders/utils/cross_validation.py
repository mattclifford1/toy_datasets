from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from data_loaders.utils.shuffling import RANDOM_STATE


def subset_rows(data: dict[str, Any], inds: np.ndarray) -> dict[str, Any]:
    '''
    build a new data dict from the given row indices

    Every per-instance array is subset together - not just 'X' and 'y' - so
    extras such as 'cost_matrix' stay row-aligned. Non-row entries
    ('description', 'feature_names', ...) are carried over untouched.
        data: data dict holder (not modified)
        inds: row indices to keep

    returns: a new data dict containing only those rows
    '''
    inds = np.asarray(inds, dtype=int)
    instances = data['X'].shape[0]
    subset: dict[str, Any] = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray) and val.shape[0] == instances:
            subset[key] = val[inds]
        else:
            subset[key] = val
    return subset


def stratified_kfold_split(
        data: dict[str, Any],
        n_splits: int = 5,
        seed: bool | int = True,
        shuffle: bool = True,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    '''
    split a data dict into k cross validation folds, preserving class ratios

    Each fold is used once as the validation set, with the remaining k-1 folds
    forming the training set. Class proportions are held as close to the full
    dataset as the fold size allows, which matters for imbalanced data where an
    unstratified fold can easily contain no minority instances at all.
        data: data dict holder (not modified)
        n_splits: number of folds, k
        seed: random seed (True means default seed, False means non-deterministic,
            int means use that value)
        shuffle: shuffle instances within each class before splitting

    returns: list of (train_split, val_split) data dict pairs, length n_splits

    raises: ValueError if n_splits < 2 or a class has fewer than n_splits members
    '''
    if n_splits < 2:
        raise ValueError(f'n_splits needs to be at least 2 instead of :{n_splits}')

    y = np.asarray(data['y'])
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < n_splits:
        raise ValueError(
            f'cannot make {n_splits} stratified folds: class '
            f'{classes[np.argmin(counts)]} has only {counts.min()} instances'
        )

    fold_inds = _stratified_fold_indices(y, n_splits=n_splits, seed=seed, shuffle=shuffle)
    all_inds = np.arange(len(y))

    folds: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for val_inds in fold_inds:
        train_inds = np.setdiff1d(all_inds, val_inds)
        folds.append((subset_rows(data, train_inds), subset_rows(data, val_inds)))
    return folds


def stratified_kfold_indices(
        y: np.ndarray,
        n_splits: int = 5,
        seed: bool | int = True,
        shuffle: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    '''
    yield (train_inds, val_inds) for stratified k-fold, without touching the data

    Use this instead of :func:`stratified_kfold_split` when the folds index into
    something other than a data dict, or when the caller wants to resample the
    rows itself before building the splits.
        y: label array of shape (n_samples,)
        n_splits: number of folds, k
        seed: random seed (see :func:`stratified_kfold_split`)
        shuffle: shuffle instances within each class before splitting

    yields: (train_inds, val_inds) index arrays, n_splits times
    '''
    y = np.asarray(y)
    fold_inds = _stratified_fold_indices(y, n_splits=n_splits, seed=seed, shuffle=shuffle)
    all_inds = np.arange(len(y))
    for val_inds in fold_inds:
        yield np.setdiff1d(all_inds, val_inds), val_inds


def _stratified_fold_indices(
        y: np.ndarray,
        n_splits: int,
        seed: bool | int,
        shuffle: bool,
) -> list[np.ndarray]:
    '''
    assign every row to one of n_splits folds, class by class

    Each class's rows are dealt round-robin across the folds, so the remainder
    is spread out rather than piling into the last fold - with 12 minority rows
    and 5 folds that gives fold sizes 3,3,2,2,2 rather than 2,2,2,2,4.
    '''
    rng = np.random.default_rng(_seed_value(seed))
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        cls_inds = np.where(y == cls)[0]
        if shuffle:
            rng.shuffle(cls_inds)
        for fold, chunk in enumerate(np.array_split(cls_inds, n_splits)):
            folds[fold].extend(chunk.tolist())
    return [np.sort(np.asarray(fold, dtype=int)) for fold in folds]


def _seed_value(seed: bool | int) -> int | None:
    '''map the package's bool/int seed convention onto a numpy generator seed'''
    if seed is True:
        return RANDOM_STATE
    if seed is False:
        return None
    return int(seed)
