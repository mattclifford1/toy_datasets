# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for cervical cancer dataset: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors
This dataset focuses on the prediction of indicators/diagnosis of cervical cancer. The features cover demographic information, habits, and historic medical records.
'''

import os
import pandas as pd
from data_loaders.abstract_loader import AbstractLoader


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class cervical_cancer_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.7,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size,
                         dataset_name='Cervical Cancer',
                         **kwargs)

    def load_data(self):
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                        'datasets', 'cervical_cancer', 'data.csv'))
        #TODO: sort out the missing values - remove or imput?

        #TODO: which col to use as class?
        data['y'] = df.pop('????').to_numpy()

        data['X'] = df.to_numpy()
        #TODO: which cols to use as features? (there might be multiple targets)
        data['feature_names'] = df.columns.to_list()
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', 
                            'datasets', 'cervical_cancer', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        return data


if __name__ == "__main__":
    loader = cervical_cancer_loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()