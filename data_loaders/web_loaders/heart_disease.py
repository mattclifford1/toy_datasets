# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for hepititus: https://archive.ics.uci.edu/dataset/46/hepatitis
'''

import os
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from data_loaders.abstract_loader import AbstractLoader


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class heart_disease_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.5,
                #  split_ratio=10,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                        #  split_ratio=split_ratio, 
                         dataset_name='Heart Disease',
                         **kwargs)
        
    def load_data(self):
        data = {}
        # df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
        #                  'datasets', 'hepititus', 'data.csv'))

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
        return data
        

if __name__ == "__main__":
    loader = heart_disease_loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()
