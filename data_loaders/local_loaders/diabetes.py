# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for the Pima Indians Diabetes Database: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download
'''

import os
import pandas as pd
from data_loaders.abstract_loader import AbstractLoader


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class diabetes_pima_indians_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.5,
                 split_ratio=10,
                 **kwargs):
        # size=0.2, ratio=5   works nicely
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                         split_ratio=split_ratio, 
                         dataset_name='Diabetes Pima Indians',
                         **kwargs)
        
    def load_data(self):
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                        'datasets', 'diabetes_pima_indians', 'data.csv'))
        
        data['y'] = df.pop('Outcome').to_numpy()

        # keep:            glucose, BMI, age, insulin, and skin thickness
        # maybe keep only: glucose, BMI, age
        # according to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8943493/ 
        remove_cols = [
            # 'Pregnancies',
            # 'DiabetesPedigreeFunction', 
            # 'BloodPressure', 
        #    'Insulin',
        #    'SkinThickness'
                    ]
        for col in remove_cols:
            df.pop(col)
        data['X'] = df.to_numpy()
        data['feature_names'] = df.columns.to_list()
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'diabetes_pima_indians', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        return data

    
if __name__ == "__main__":
    loader = diabetes_pima_indians_loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()