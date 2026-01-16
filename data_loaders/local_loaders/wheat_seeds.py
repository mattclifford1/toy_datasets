# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Wheat seed type predcition dataset (3 classes)
UCI dataset: https://archive.ics.uci.edu/ml/datasets/seeds#
instances: 210
attributes: 7
'''

import os
import pandas as pd
from data_loaders.abstract_loader import AbstractLoader


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class wheat_seeds_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.7,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                         dataset_name='Wheat Seeds',
                         **kwargs)
        
    def load_data(self):
        '''
        we remove class 3 and make it a binary problem
        '''
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                        'datasets', 'wheat_seeds', 'data.csv'), header=None)
        df.drop(df[df[7] == 3].index, inplace=True)
        df = df.replace({7: {2: 0}})
        data['y'] = df.pop(7).to_numpy()  # type: ignore
        data['X'] = df.to_numpy()
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'wheat_seeds', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        data['feature_names'] = ['area',
                                'perimeter',
                                'compactness',
                                'length of kernel',
                                'width of kernel',
                                'asymmetry coefficient',
                                'length of kernel groove']
        return data


if __name__ == "__main__":
    loader = wheat_seeds_loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()