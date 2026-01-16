# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Get datasets from the costcla package
    - credit scoring and direct marketing
'''

import os
import pandas as pd
from data_loaders.abstract_loader import AbstractLoader


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class costcla_CreditScoring_Kaggle2011_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.7,
                 split_ratio=10,
                 percent_of_data=5,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                         split_ratio=split_ratio, 
                         dataset_name='CreditScoring_Kaggle2011_costcla',
                         set_seed=True,
                         percent_of_data=percent_of_data,
                         **kwargs)
        
    def load_data(self):
        return _get_costcla_dataset('CreditScoring_Kaggle2011_costcla')


class costcla_CreditScoring_PAKDD2009_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.7,
                 split_ratio=10,
                 percent_of_data=5,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                         split_ratio=split_ratio, 
                         dataset_name='CreditScoring_PAKDD2009_costcla',
                         set_seed=True,
                         percent_of_data=percent_of_data,
                         **kwargs)
        
    def load_data(self):
        return _get_costcla_dataset('CreditScoring_PAKDD2009_costcla')
    

class costcla_DirectMarketing_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.7,
                 split_ratio=10,
                 percent_of_data=5,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                         split_ratio=split_ratio, 
                         dataset_name='DirectMarketing_costcla',
                         set_seed=True,
                         percent_of_data=percent_of_data,
                         **kwargs)
        
    def load_data(self):
        return _get_costcla_dataset('DirectMarketing_costcla')


def _get_costcla_dataset(dataset="CreditScoring_Kaggle2011_costcla", normalise=False):
    '''
    load the costcla csv dataset files
    available datasets:
        - CreditScoring_Kaggle2011_costcla
        - CreditScoring_PAKDD2009_costcla
        - DirectMarketing_costcla
    '''
    data = {}
    csvs = ['X', 'y', 'cost_matrix']
    # read and store all csv data
    for csv in csvs:
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..', 'datasets', dataset, f'{csv}.csv'))
        # split into train and test
        data[csv] = df.to_numpy()
        if data[csv].shape[1] == 1:
            data[csv] = data[csv].ravel()
        # get feature names
        if csv == 'X':
            data['feature_names'] = df.columns.to_list()

    # normalise X data
    if normalise == True:
        data['X'] = data['X'] / data['X'].max(axis=0)

    # add  description
    with open(os.path.join(CURRENT_FILE, '..', 'datasets', dataset, 'description.txt'), 'r') as f:
        data['description'] = f.read()
    return data


if __name__ == '__main__':
    loader = costcla_CreditScoring_Kaggle2011_loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()
