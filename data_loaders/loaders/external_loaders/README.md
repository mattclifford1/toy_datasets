# data_loaders.loaders.external_loaders

Medical ICU datasets requiring special data access.

## Overview

These loaders target MIMIC-III and MIMIC-IV datasets from PhysioNet. The CSV
files are **not** included in the repository due to MIMIC licensing
restrictions — you must download and preprocess the data yourself.

The loaders look for the CSVs in this order:

1. an explicit `data_path` argument passed to the loader,
2. the `MIMIC_DATA_DIR` environment variable,
3. the default `~/datasets`.

The resolved directory must contain `MIMIC-III/` and/or `MIMIC-IV/` sub-folders
holding the preprocessed CSVs. If a file is missing the loader raises a
`FileNotFoundError` with download instructions.

## API

These datasets are registered like any other and can be loaded by name:

| Registered name | Loader class | Task |
| --- | --- | --- |
| `MIMIC-III Mortality` | `MIMICIIIMortalityLoader` | mortality (death/readmission vs discharge) |
| `MIMIC-III Sepsis` | `MIMICIIISepsisLoader` | sepsis label (2019 challenge) |
| `MIMIC-IV Ready for Discharge` | `MIMICIVReadyForDischargeLoader` | ready-for-discharge |

`MIMIC-III Mortality` accepts `complete=False` (default, imputed data) or
`complete=True` (complete-case data).

For backwards compatibility, function-based helpers returning a pre-split
`(train_dict, test_dict)` tuple are still available:
`MIMIC_III.get_mortality(seed, complete, data_path)`,
`MIMIC_III.get_sepsis(seed, data_path)`, and
`MIMIC_IV.get_ready_for_discharge(seed, data_path)`.

## Data access

MIMIC data is available through PhysioNet after completing a credentialed
access application:
- [PhysioNet MIMIC-III](https://physionet.org/content/mimiciii/)
- [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/)

Or email Matt Clifford <matt.clifford@bristol.ac.uk> for the preprocessed CSVs.

## Usage

```python
from data_loaders import get_dataset

# Uses the default path, MIMIC_DATA_DIR, or an explicit data_path=...
loader = get_dataset('MIMIC-III Mortality', data_path='~/path/to/mimic/data')

X, y = loader.get_X(), loader.get_y()
train, test = loader.get_train_test_split()
X_train, y_train = train['X'], train['y']
X_test,  y_test  = test['X'],  test['y']
```
