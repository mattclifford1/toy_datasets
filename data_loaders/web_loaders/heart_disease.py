# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
https://archive.ics.uci.edu/dataset/45/heart+disease
'''
from __future__ import annotations

from typing import Any

import numpy as np
from ucimlrepo import fetch_ucirepo
from data_loaders.abstract_loader import AbstractLoader, DataDict


class heart_disease_loader(AbstractLoader):
    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                #  minority_reduce_scaler=10,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                        #  minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='Heart Disease',
                         **kwargs)

    def load_data(self) -> DataDict:
        data = {}

        # fetch dataset
        heart_disease = fetch_ucirepo(id=45)

        # data (as pandas dataframes)
        X = heart_disease.data.features
        y = heart_disease.data.targets

        X.pop('ca')
        X.pop('thal')

        data['y'] = y.to_numpy().squeeze(axis=1)
        data['X'] = X.to_numpy()

        for i in [1,2,]:
            inds = np.where(data['y']==i)
            data['y'] = np.delete(data['y'], inds)
            data['X'] = np.delete(data['X'], inds, axis=0)
        data['y'][data['y']==2] = 1
        data['y'][data['y']==3] = 1
        data['y'][data['y']==4] = 1

        data['feature_names'] = heart_disease.metadata
        # add name and description
        data['description'] = heart_disease.metadata
        data['label_names'] = ['no heart disease', 'heart disease']
        return data
        

if __name__ == "__main__":
    loader = heart_disease_loader()
    print(loader.get_info(long=True))
    # loader.plot_dataset()
    loader.plot_train_test_split()
