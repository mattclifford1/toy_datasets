# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for the Statlog German Credit dataset
LINKS:
    - https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    - https://www.kaggle.com/datasets/uciml/german-credit

Credit risk: 1000 applicants described by 20 attributes, labelled good (700) or bad (300)
credit risk. Notable for coming with an official asymmetric cost matrix - classifying a
bad customer as good is stated to be five times worse than the reverse - which makes it a
standard benchmark for cost sensitive classification.
'''
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from data_loaders.utils import binarise_labels
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict

# 13 of the 20 attributes are stored as codes 'A11', 'A34', 'A201' ... where the digits
# after the attribute number order the levels (A11 < A12 < A13 < A14). We keep the
# trailing index as an ordinal value: see the note on encoding in the class docstring.
_CODE = re.compile(r'^A(\d+)$')


def _encode_codes(column: pd.Series) -> np.ndarray:
    '''map the UCI 'Annn' level codes of one attribute to 0-based ordinals'''
    levels = sorted(column.dropna().unique(), key=lambda c: int(_CODE.match(c).group(1)))
    lookup = {code: i for i, code in enumerate(levels)}
    return column.map(lookup).to_numpy(dtype=float)


class GermanCreditLoader(AbstractLoader):
    """Load the Statlog German Credit dataset.

    Binary classification of credit risk: good (class 0, 700 cases) against bad
    (class 1, 300 cases), from 20 attributes covering account status, credit
    history, employment, personal status and property.

    Dataset stats: 1000 samples, 20 features, 30% minority class.
    Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

    Notes
    -----
    Thirteen of the twenty attributes are categorical, stored by UCI as codes of
    the form ``A11``, ``A34``, ``A201``. They are encoded here as 0-based
    **ordinals**, ordered by the trailing index, which keeps all 20 attributes
    and their names intact.

    This is a deliberate simplification and worth knowing about. For attributes
    that really are ordered - checking account status, savings, length of
    employment - it is faithful. For genuinely nominal ones such as *Purpose*
    (car, furniture, radio/TV, ...) it imposes an order that does not exist, so a
    model may read a meaningless magnitude into it. The alternatives are worse
    for this package's purposes: one-hot encoding turns 20 named features into
    roughly 60 anonymous ones, and the UCI-supplied ``german.data-numeric``
    variant drops the attribute names entirely. Use
    ``german_credit_numeric.csv`` directly if you need the official numeric
    encoding.

    The dataset's published cost matrix - misclassifying a bad risk as good costs
    5, the reverse costs 1 - is not applied to the labels here; it is recorded in
    ``description`` for anyone doing cost sensitive work.

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.7
        Fraction of data used for training in train/test splits.
    minority_reduce_scaler : int or None, default=None
        If set, reduce the minority class in the train set by this factor.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.7,
                 minority_reduce_scaler: int | None = None,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='German Credit',
                         short_description='Credit risk from 20 applicant attributes, with an official 5:1 cost matrix — binary, imbalanced',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Load the Statlog German Credit CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        data = {}
        df = pd.read_csv(self.local_dataset_path('german_credit', 'german_credit.csv'),
                         index_col=0)
        # 1 -> good credit risk (0), 2 -> bad credit risk (1), so the minority
        # class is 1, consistent with the other imbalanced loaders here
        data['y'] = binarise_labels(df.pop('label'), {1: 0, 2: 1})

        for column in df.columns:
            if df[column].map(lambda v: bool(_CODE.match(str(v)))).all():
                df[column] = _encode_codes(df[column].astype(str))

        data['X'] = df.to_numpy().astype(float)
        data['feature_names'] = df.columns.to_list()
        data['label_names'] = ['good credit risk', 'bad credit risk']
        data['description'] = self.local_dataset_description('german_credit')
        return data


if __name__ == "__main__":
    loader = GermanCreditLoader()
    print(loader.get_info(long=True))
    # loader.plot_dataset()
    loader.plot_train_test_split()
