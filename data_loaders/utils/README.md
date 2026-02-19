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

**`proportional_split(data, train_size, seed, minority_reduce_scaler, equal_test, minority_reduce_scaler_test)`**
Split a data dict into train and test sets while preserving class
distributions. Supports optional minority-class reduction in both splits and
test-set balancing via `equal_test`.

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
