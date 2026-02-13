# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for Habermans breast cancer: https://archive.ics.uci.edu/dataset/43/haberman+s+survival
The dataset contains cases from a study that was conducted between 1958 and 1970 at the
University of Chicago's Billings Hospital on the survival of patients who had undergone surgery
for breast cancer.
'''
from __future__ import annotations

import os
from typing import Any

import pandas as pd
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class HabermansBreastCancerLoader(AbstractLoader):
    """Load Haberman's Survival dataset.

    Binary classification predicting whether a breast-cancer surgery patient
    survived 5 or more years (class 0) or died within 5 years (class 1).
    Study conducted 1958–1970 at the University of Chicago Billings Hospital.

    Dataset stats: 306 samples, 3 features (age, year of surgery, positive nodes).
    Source: https://archive.ics.uci.edu/dataset/43/haberman+s+survival

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
                         dataset_name='Habermans Breast Cancer',
                         **kwargs)
        
    def load_data(self) -> DataDict:
        """Load the Haberman's Survival CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..', '..', 
                        'datasets', 'Habermans_breast_cancer', 'data.csv'))
        data['y'] = df.pop('Survived_Longer_5_Years').to_numpy()
        data['y'][data['y']==1] = 0  
        data['y'][data['y']==2] = 1
        data['X'] = df.to_numpy()
        data['feature_names'] = df.columns.to_list()
        data['label_names'] = ['survived 5 years or longer', 'died within 5 year']
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', '..', 'datasets', 'Habermans_breast_cancer', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        return data


if __name__ == "__main__":
    loader = HabermansBreastCancerLoader()
    print(loader.get_info(long=True))
    # loader.plot_dataset()
    loader.plot_train_test_split()
