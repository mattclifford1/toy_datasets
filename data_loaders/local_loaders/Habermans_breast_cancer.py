# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for Habermans breast cancer: https://archive.ics.uci.edu/dataset/43/haberman+s+survival
The dataset contains cases from a study that was conducted between 1958 and 1970 at the 
University of Chicago's Billings Hospital on the survival of patients who had undergone surgery 
for breast cancer.
'''

import os
import pandas as pd
from data_loaders.abstract_loader import AbstractLoader


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class habermans_breast_cancer_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.7,
                 split_ratio=None,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                         split_ratio=split_ratio, 
                         dataset_name='Habermans Breast Cancer',
                         **kwargs)
        
    def load_data(self):
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                        'datasets', 'Habermans_breast_cancer', 'data.csv'))
        data['y'] = df.pop('Survived_Longer_5_Years').to_numpy()
        data['y'][data['y']==1] = 0  
        data['y'][data['y']==2] = 1
        data['X'] = df.to_numpy()
        data['feature_names'] = df.columns.to_list()
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'Habermans_breast_cancer', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        return data


if __name__ == "__main__":
    loader = habermans_breast_cancer_loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()
