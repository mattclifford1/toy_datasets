# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
abalone gender UCI dataset: https://archive.ics.uci.edu/ml/datasets/Abalone
instances: 4177
'''
import os
import pandas as pd
from data_loaders.abstract_loader import AbstractLoader


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class abalone_gender_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.5,
                 split_ratio=10,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                         split_ratio=split_ratio, 
                         dataset_name='Abalone Gender',
                         **kwargs)
        
    def load_data(self):
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                        'datasets', 'abalone', 'data.csv'), header=None)
        df.drop(df[df[0] == 'I'].index, inplace=True)
        df = df.replace({0: {'M': 0, 'F': 1}})
        data['y'] = df.pop(0).to_numpy()  # type: ignore
        data['X'] = df.to_numpy()
        data['feature_names'] = ['Length',
                                'Diameter',
                                'Height',
                                'Whole weight',
                                'Shucked weight',
                                'Viscera weight',
                                'Shell weight',
                                'Rings']
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'abalone', 'description.txt'), 'r') as f:
            data['description'] = f.read()

        return data
    

if __name__ == "__main__":
    loader = abalone_gender_loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()
