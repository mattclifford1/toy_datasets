# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for chronic kidney disease dataset: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
This dataset can be used to predict the chronic kidney disease and it can be collected from the hospital nearly 2 months of period.
'''

import os
import pandas as pd
from data_loaders.abstract_loader import AbstractLoader


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class chronic_kidney_disease_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.7,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                         dataset_name='Chronic Kidney Disease',
                         **kwargs)
        
    def load_data(self):
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                        'datasets', 'chronic_kidney_disease', 'data.csv'))
        #TODO: sort out the missing values - remove or imput?

        data['y'] = df.pop('class').to_numpy()
        # classes are ?, notpresent, notckd and ckd (chronic kidney disease) 
        # take as ckd or not ckd for ML algorithms
        #TODO: process 'class'

        data['X'] = df.to_numpy()
        data['feature_names'] = df.columns.to_list()
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', 
                            'datasets', 'chronic_kidney_disease', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        # shuffle the dataset
        return data


if __name__ == "__main__":
    loader = chronic_kidney_disease_loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()