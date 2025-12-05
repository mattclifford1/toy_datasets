# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Ionosphere Dataset requires the prediction of structure in the atmosphere given radar returns targeting free electrons in the ionosphere.
UCI dataset: https://archive.ics.uci.edu/ml/datasets/Ionosphere
instances: 351
attributes: 34
'''
import os
import pandas as pd
from data_loaders.abstract_loader import AbstractLoader


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class ionosphere_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.7,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                         dataset_name='Ionosphere',
                         **kwargs)
        
    def load_data(self):
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                        'datasets', 'Ionosphere', 'data.csv'), header=None)
        df.drop(df[df[0] == 'I'].index, inplace=True)
        df = df.replace({34: {'b': 0, 'g': 1}})
        data['y'] = df.pop(34).to_numpy()  # type: ignore
        data['X'] = df.to_numpy()
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'Ionosphere', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        return data


if __name__ == "__main__":
    loader = ionosphere_loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()
