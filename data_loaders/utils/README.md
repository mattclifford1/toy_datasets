# data_loaders.utils

Data utilities for normalization, shuffling, seeding, and train/test splitting.

## Overview

This sub-package provides the low-level building blocks used by all loaders: a
fitted MinMax scaler, reproducible shuffling helpers, and a class-aware
train/test splitter.

## Classes and Functions

**`Normaliser(train_X)`**
MinMax scaler fitted on training data. Scales all features to the range
`[-1, 1]`. Call the instance to transform a feature matrix; use
`transform_instance()` for a single row.

**`proportional_split(data, train_size, seed, minority_reduce_scaler, equal_test, minority_reduce_scaler_test, majority_max)`**
Split a data dict into train and test sets while preserving class
distributions. Supports optional minority-class reduction in both splits,
test-set balancing via `equal_test`, and capping the train-split majority via
`majority_max` (applied first, so ratios are taken against the capped count).
Every per-instance array in the dict — not just `X` and `y` — is split and
trimmed together, so extras such as `cost_matrix` stay row-aligned.

**`stratified_kfold_split(data, n_splits, seed, shuffle)`**
Split a data dict into `n_splits` cross validation folds, returning a list of
`(train_split, val_split)` pairs. Class ratios are preserved in every fold — on
imbalanced data an unstratified fold can easily contain no minority instances
at all. Per-instance arrays such as `cost_matrix` are split with their rows.

**`stratified_kfold_indices(y, n_splits, seed, shuffle)`**
The same folds as index arrays, yielding `(train_inds, val_inds)`. Use it when
the folds index into something other than a data dict, or when the caller wants
to resample the rows itself before building the splits.

**`subset_rows(data, inds)`**
Build a new data dict from the given row indices, subsetting every per-instance
array together and carrying non-row entries over untouched.

**`shuffle_data(data, seed)`**
Shuffle `X` and `y` in unison using sklearn. Returns the modified data dict.

**`shuffle_dataset(data, seed)`**
Like `shuffle_data` but applies the same permutation to every numpy array in
the dict whose first dimension matches the number of instances (e.g. `costs`).

**`set_seed(seed)`**
Set the NumPy random seed. Pass `True` for the default seed (42), `False` for
non-deterministic, or an integer for a custom seed.

**`RANDOM_STATE`**
Default random seed constant (`42`).

## Usage

```python
from data_loaders.utils import Normaliser, proportional_split

data = {'X': X_all, 'y': y_all}

# Split first, then fit scaler on train only
train, test = proportional_split(data, train_size=0.8)

scaler = Normaliser(train['X'])
train['X'] = scaler(train['X'])
test['X']  = scaler(test['X'])
```
