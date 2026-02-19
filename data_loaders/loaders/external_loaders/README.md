# data_loaders.loaders.external_loaders

Medical ICU datasets requiring special data access.

## Overview

These loaders target MIMIC-III and MIMIC-IV datasets from PhysioNet. The CSV
files are not included in the repository due to MIMIC licensing restrictions.
You must download and preprocess the data yourself, then place it in
`data_loaders/datasets/MIMIC-III/` or `data_loaders/datasets/MIMIC-IV/`.

## API

Unlike other loaders these are function-based rather than class-based. Each
function returns a `(train_dict, test_dict)` tuple pre-split at 50/50.

### MIMIC-III (`data_loaders.loaders.external_loaders.MIMIC_III`)

**`get_mortality(seed, complete) -> (train_dict, test_dict)`**
ICU mortality prediction.
- Label `0` — successful discharge.
- Label `1` — death or readmission.
- `complete=False` (default) uses imputed data; `complete=True` uses
  complete-case data.

**`get_sepsis(seed) -> (train_dict, test_dict)`**
Sepsis label prediction from the MIMIC 2019 challenge.

### MIMIC-IV (`data_loaders.loaders.external_loaders.MIMIC_IV`)

**`get_ready_for_discharge(seed) -> (train_dict, test_dict)`**
Ready-for-discharge prediction from real-time ICU observations.

## Data access

MIMIC data is available through PhysioNet after completing a credentialed
access application:
- [PhysioNet MIMIC-III](https://physionet.org/content/mimiciii/)
- [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/)

## Usage

```python
from data_loaders.loaders.external_loaders import MIMIC_III

train, test = MIMIC_III.get_mortality(seed=42, complete=False)

X_train, y_train = train['X'], train['y']
X_test,  y_test  = test['X'],  test['y']
```
