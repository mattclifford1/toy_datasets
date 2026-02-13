# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for cervical cancer dataset: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors
This dataset focuses on the prediction of indicators/diagnosis of cervical cancer. The features cover demographic information, habits, and historic medical records.
'''
from __future__ import annotations

import os
from typing import Any

import pandas as pd
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class CervicalCancerLoader(AbstractLoader):
    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.7,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Cervical Cancer',
                         **kwargs)

    def load_data(self) -> DataDict:
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..', '..', 
                        'datasets', 'cervical_cancer', 'data.csv'))
        #TODO: sort out the missing values - remove or imput?

        #TODO: which col to use as class?
        data['y'] = df.pop('????').to_numpy()

        data['X'] = df.to_numpy()
        #TODO: which cols to use as features? (there might be multiple targets)
        data['feature_names'] = df.columns.to_list()
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', '..', 
                            'datasets', 'cervical_cancer', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        return data


if __name__ == "__main__":
    loader = CervicalCancerLoader()
    # loader.plot_dataset()
    loader.plot_train_test_split()