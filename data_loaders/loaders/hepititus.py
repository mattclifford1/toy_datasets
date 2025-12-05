# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for hepititus: https://archive.ics.uci.edu/dataset/46/hepatitis
'''

import os
import pandas as pd
from data_loaders.abstract_loader import AbstractLoader


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class hepititus_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.5,
                #  split_ratio=5,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                        #  split_ratio=split_ratio, 
                         dataset_name='Hepatitis',
                         **kwargs)
        
    def load_data(self):
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                        'datasets', 'hepititus', 'data.csv'))
        df.pop('PROTIME')
        df.pop('ALKPHOSPHATE')
        df.pop('ALBUMIN')
        # df.pop('LIVERBIG')
        # df.pop('LIVERFIRM')
        df = df[~df.isin(['?']).any(axis=1)]
        data['y'] = df.pop('Class').to_numpy()
        data['y'][data['y'] == 2] = 0
        data['X'] = df.to_numpy()
        data['feature_names'] = df.columns.to_list()
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'hepititus', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        return data
        

if __name__ == "__main__":
    loader = hepititus_loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()