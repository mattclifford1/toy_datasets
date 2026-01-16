# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
banknote authentication UCI dataset: https://archive.ics.uci.edu/ml/datasets/banknote+authentication#
instances: 1372
'''

import os
import pandas as pd
from data_loaders.abstract_loader import AbstractLoader

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class banknote_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.7,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size,
                         dataset_name='Banknote Authentication',
                         **kwargs)

    def load_data(self):
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                        'datasets', 'banknote_authentication', 'data.csv'), header=None)
        data['y'] = df.pop(4).to_numpy()  # type: ignore
        data['X'] = df.to_numpy()
        data['feature_names'] = ['variance of Wavelet Transformed image',
                                'skewness of Wavelet Transformed image',
                                'curtosis of Wavelet Transformed image',
                                'entropy of image']
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'banknote_authentication', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        return data
    
    
if __name__ == "__main__":
    loader = banknote_loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()