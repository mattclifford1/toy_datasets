'''
loaders for MIMIC-III ICU tasks (mortality and sepsis).

N.B. MIMIC data is not shipped due to PhysioNet licensing - you must download
and preprocess it yourself (or email Matt for help). By default the loaders look
for the CSVs under ``~/datasets/MIMIC-III``; override with
the ``data_path`` argument or the ``MIMIC_DATA_DIR`` environment variable.

Mortality task:
    - 1 is negative outcome (death or readdmission)
    - 0 successful discharge
read the paper for full details: https://bmjopen.bmj.com/content/bmjopen/9/3/e025925.full.pdf
and the github repo https://github.com/UHBristolDataScience/smartt-algortihm/tree/main
processing https://github.com/UHBristolDataScience/towards-decision-support-icu-discharge

Resampled data we exclude IMPUTED:
    row 2 to 784 (outcome 0): original NRFD
    row 785 to 6606 (outcome 0): resampled NRFD (again should be in blocks of monotonically increasing ICUSTAY_ID, of which you could take one block).
    row 6607 to 13244 (outcome 1): RFD

Resampled data we exclude COMPLETECASE:
    row 2 to 2508: RFD
    row 2509 to 2837: original NRFD
    row 2838 onwards: resampled NRFD

'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
from __future__ import annotations

from typing import Any

import pandas as pd

from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict
from data_loaders.loaders.external_loaders import mimic_file


class MIMICIIIMortalityLoader(AbstractLoader):
    """Load the MIMIC-III ICU mortality task.

    Binary classification: death or readmission (class 1) vs. successful
    discharge (class 0), predicted from ICU feature measurements.

    MIMIC data is not shipped (PhysioNet licensing). See the module docstring
    for how to provide it.

    Parameters
    ----------
    data_path : str or None, default=None
        Base directory containing the ``MIMIC-III/`` folder. When None, falls
        back to the ``MIMIC_DATA_DIR`` env var then the packaged default.
    complete : bool, default=False
        If False use the imputed CSV, if True use the complete-case CSV.
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 data_path: str | None = None,
                 complete: bool = False,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='MIMIC-III Mortality',
                         **kwargs)
        self.data_path = data_path
        self.complete = complete

    def load_data(self) -> DataDict:
        """Load and preprocess the MIMIC-III mortality CSV.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``.
        """
        data: dict[str, Any] = {}
        if self.complete is False:
            file = 'fm_MIMIC_IMPUTED_extended.csv'
        else:
            file = 'fm_MIMIC_COMPLETECASE_extended.csv'
        df = pd.read_csv(mimic_file('MIMIC-III', file, self.data_path))

        # remove resampled data we dont care about for the true dataset
        if self.complete is False:
            df.drop(df.index[0:2], inplace=True)
            df.drop(df.index[785:6606], inplace=True)
        else:
            df.drop(df.index[0:2], inplace=True)
            df.drop(df.index[2837:], inplace=True)

        # not a feature
        df.pop('cohort')
        # not useful features according to paper
        df.pop('age')
        df.pop('sex')
        df.pop('bmi')
        df.pop('los')

        # label
        data['y'] = df.pop('outcome').to_numpy().copy()
        # swap the inds as use convention of minority being 1
        # data orignally is 0 is negative outcome (death or readdmission) - 1 successful discharge
        # but we change that to be the opposite
        data['y'][data['y'] == 1] = 2
        data['y'][data['y'] == 0] = 1
        data['y'][data['y'] == 2] = 0

        # features
        data['X'] = df.to_numpy()
        data['feature_names'] = df.columns.to_list()
        data['label_names'] = ['Successful Discharge', 'Death or Readmission']
        return data


class MIMICIIISepsisLoader(AbstractLoader):
    """Load the MIMIC-III sepsis-prediction task (2019 challenge).

    Binary classification of the sepsis label from ICU measurements.
    Data: https://www.kaggle.com/datasets/missan/mimic-challenge-2019

    MIMIC data is not shipped (PhysioNet licensing). See the module docstring
    for how to provide it.

    Parameters
    ----------
    data_path : str or None, default=None
        Base directory containing the ``MIMIC-III/`` folder. When None, falls
        back to the ``MIMIC_DATA_DIR`` env var then the packaged default.
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 data_path: str | None = None,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='MIMIC-III Sepsis',
                         **kwargs)
        self.data_path = data_path

    def load_data(self) -> DataDict:
        """Load the MIMIC-III sepsis CSV.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``.
        """
        data: dict[str, Any] = {}
        df = pd.read_csv(mimic_file(
            'MIMIC-III', 'mimic_challenge_2019_sepsis.csv', self.data_path))

        # not a feature
        df.pop('ID')

        # label
        data['y'] = df.pop('SepsisLabel').to_numpy()

        # features
        data['X'] = df.to_numpy()
        data['feature_names'] = df.columns.to_list()
        data['label_names'] = ['No Sepsis', 'Sepsis']
        return data


# ---------------------------------------------------------------------------
# Backwards-compatible function API (returns pre-split (train, test) dicts)
# ---------------------------------------------------------------------------
def get_mortality(
        seed: bool | int = True,
        complete: bool = False,
        data_path: str | None = None,
        **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Backwards-compatible helper: return a (train, test) split for mortality."""
    loader = MIMICIIIMortalityLoader(
        data_path=data_path, complete=complete, set_seed=seed, **kwargs)
    return loader.get_train_test_split(seed=seed)


def get_sepsis(
        seed: bool | int = True,
        data_path: str | None = None,
        **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Backwards-compatible helper: return a (train, test) split for sepsis."""
    loader = MIMICIIISepsisLoader(data_path=data_path, set_seed=seed, **kwargs)
    return loader.get_train_test_split(seed=seed)
