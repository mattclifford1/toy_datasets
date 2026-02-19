# data_loaders.resampling

Class-imbalance resampling strategies: upsampling, SMOTE, and downsampling.

## Overview

All resamplers share a `__call__(X, y) -> (X, y)` interface and inherit from
`AbstractResampler`. The functional helpers (`stratified_subsample`,
`proportional_downsample`) operate directly on numpy arrays or data dicts.

## Classes and Functions

**`AbstractResampler`**
Abstract base class. Subclass and implement `__call__(X, y)` to create a
custom resampler.

**`RandomDuplicateMinorityUpsampler(random_state, factor)`**
Upsample the minority class by duplicating rows at random.
- `factor='equal'` (default) — upsample to match the majority class count.
- `factor=<float > 1>` — upsample by that multiplier.

**`SMOTEUpsampler(random_state, k_neighbors)`**
Generate synthetic minority-class samples via SMOTE. Requires
`imbalanced-learn` (`pip install imbalanced-learn`).

**`StratifiedSubsampler(n_samples, random_state)`**
Subsample to exactly `n_samples` rows while preserving class proportions.

**`stratified_subsample(X, y, n_samples, random_state)`**
Functional wrapper around `StratifiedSubsampler`.

**`proportional_downsample(data, percent_of_data, seed)`**
Downsample a data dict to `percent_of_data`% of its rows, keeping class
proportions intact. Operates in-place on all numpy row arrays in the dict.

## Usage

```python
from data_loaders.resampling import RandomDuplicateMinorityUpsampler, SMOTEUpsampler, proportional_downsample

# Duplicate minority rows until classes are balanced
upsampler = RandomDuplicateMinorityUpsampler(factor='equal')
X_bal, y_bal = upsampler(train['X'], train['y'])

# Generate synthetic minority samples with SMOTE
smote = SMOTEUpsampler(k_neighbors=5)
X_syn, y_syn = smote(train['X'], train['y'])

# Keep only 50% of data, preserving class proportions
data = proportional_downsample(data, percent_of_data=50)
```
