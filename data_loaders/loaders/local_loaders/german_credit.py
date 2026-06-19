# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Make the German credit dataset in the CLIME format
LINKS:
    - https://www.kaggle.com/datasets/uciml/german-credit
    - https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

🚧 DEV NOTE: IN CONSTRUCTION — this is a scratch script (just a ``main()`` that
previews the CSV), not yet a real loader. It is intentionally not registered in
``AVAILABLE_DATASETS`` (data_loaders/main.py) or exported from this package's
``__init__``. Outstanding work: build a ``GermanCreditLoader(AbstractLoader)``
that returns the standard ``{'X', 'y', ...}`` dict (use ``binarise_labels`` and
``local_dataset_path``/``local_dataset_description`` like the other local
loaders). Do not rely on this module until that exists.
'''
from __future__ import annotations

import os
import pandas as pd

def main() -> None:
    current_file = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(current_file, 'datasets', 'german_credit_data.csv')
    df = pd.read_csv(dataset_file)
    print(df.head())



if __name__ == '__main__':
    main()
