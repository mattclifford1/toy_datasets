# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Get datasets from the costcla package
    - credit scoring and direct marketing
'''
from __future__ import annotations

import os
from typing import Any

import pandas as pd
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class CostclaCreditScoringKaggle2011Loader(AbstractLoader):
    """Load the Kaggle 2011 Credit Scoring dataset (costcla format).

    Binary credit default classification with an associated cost matrix.
    Only 5% of the data is loaded by default to keep it manageable.

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.7
        Fraction of data used for training in train/test splits.
    minority_reduce_scaler : int, default=10
        Reduce the minority class in the train set by this factor.
    percent_of_data : float, default=5
        Percentage of the full dataset to load (0–100).
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.7,
                 minority_reduce_scaler: int = 10,
                 percent_of_data: float = 5,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='CreditScoring_Kaggle2011_costcla',
                         set_seed=True,
                         percent_of_data=percent_of_data,
                         **kwargs)
        
    def load_data(self) -> DataDict:
        """Load the Kaggle 2011 Credit Scoring costcla CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'cost_matrix'``,
            ``'feature_names'``, and ``'description'``.
        """
        return _get_costcla_dataset('CreditScoring_Kaggle2011_costcla')


class CostclaCreditScoringPAKDD2009Loader(AbstractLoader):
    """Load the PAKDD 2009 Credit Scoring dataset (costcla format).

    Binary credit default classification with an associated cost matrix.
    Only 5% of the data is loaded by default to keep it manageable.

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.7
        Fraction of data used for training in train/test splits.
    minority_reduce_scaler : int, default=10
        Reduce the minority class in the train set by this factor.
    percent_of_data : float, default=5
        Percentage of the full dataset to load (0–100).
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.7,
                 minority_reduce_scaler: int = 10,
                 percent_of_data: float = 5,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='CreditScoring_PAKDD2009_costcla',
                         set_seed=True,
                         percent_of_data=percent_of_data,
                         **kwargs)
        
    def load_data(self) -> DataDict:
        """Load the PAKDD 2009 Credit Scoring costcla CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'cost_matrix'``,
            ``'feature_names'``, and ``'description'``.
        """
        return _get_costcla_dataset('CreditScoring_PAKDD2009_costcla')


class CostclaDirectMarketingLoader(AbstractLoader):
    """Load the Direct Marketing dataset (costcla format).

    Binary classification predicting response to a direct marketing campaign,
    with an associated cost matrix for cost-sensitive learning.
    Only 5% of the data is loaded by default.

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.7
        Fraction of data used for training in train/test splits.
    minority_reduce_scaler : int, default=10
        Reduce the minority class in the train set by this factor.
    percent_of_data : float, default=5
        Percentage of the full dataset to load (0–100).
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.7,
                 minority_reduce_scaler: int = 10,
                 percent_of_data: float = 5,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='DirectMarketing_costcla',
                         set_seed=True,
                         percent_of_data=percent_of_data,
                         **kwargs)
        
    def load_data(self) -> DataDict:
        """Load the Direct Marketing costcla CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'cost_matrix'``,
            ``'feature_names'``, and ``'description'``.
        """
        return _get_costcla_dataset('DirectMarketing_costcla')


def _get_costcla_dataset(dataset: str = "CreditScoring_Kaggle2011_costcla", normalise: bool = False) -> DataDict:
    """Load a costcla dataset from local CSV files.

    Parameters
    ----------
    dataset : str, default='CreditScoring_Kaggle2011_costcla'
        Dataset subdirectory name. Available options:
        ``'CreditScoring_Kaggle2011_costcla'``,
        ``'CreditScoring_PAKDD2009_costcla'``,
        ``'DirectMarketing_costcla'``.
    normalise : bool, default=False
        If True, scale each feature column of X to [0, 1] by its maximum.

    Returns
    -------
    DataDict
        Dict with keys ``'X'``, ``'y'``, ``'cost_matrix'``,
        ``'feature_names'``, and ``'description'``.
    """
    data = {}
    csvs = ['X', 'y', 'cost_matrix']
    # read and store all csv data
    for csv in csvs:
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..', '..', 'datasets', dataset, f'{csv}.csv'))
        # split into train and test
        data[csv] = df.to_numpy()
        if data[csv].shape[1] == 1:
            data[csv] = data[csv].ravel()
        # get feature names
        if csv == 'X':
            data[csv] = data[csv].astype(float)
            data['feature_names'] = df.columns.to_list()

    # normalise X data
    if normalise == True:
        data['X'] = data['X'] / data['X'].max(axis=0)

    # add  description
    with open(os.path.join(CURRENT_FILE, '..', '..', 'datasets', dataset, 'description.txt'), 'r') as f:
        data['description'] = f.read()
    return data


if __name__ == '__main__':
    loader = CostclaCreditScoringKaggle2011Loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()
