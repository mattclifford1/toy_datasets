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
                 train_size=0.7,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Chronic Kidney Disease',
                         **kwargs)
        
    def load_data(self):
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                        'datasets', 'chronic_kidney_disease', 'data.csv'))
        # remove final row error
        df = df.drop(columns=['Unnamed: 25'])
        # drop categorical columns
        categorical_cols = [
            # 'sg', # (1.005,1.010,1.015,1.020,1.025)
            'al', # (0,1,2,3,4,5)
            'su', # (0,1,2,3,4,5)
        ]
        df = df.drop(columns=categorical_cols)

        binary_cols = [
            'rbc', # (normal,abnormal)
            'pc',  # (normal,abnormal)
            'pcc', # (present, notpresent)
            'ba', # (present, notpresent)
            'htn', # (yes, no)
            'dm', # (yes,no)
            'cad', # (yes,no)
            'appet', # (good, poor)
            'pe', # (yes,no)
            'ane' # (yes,no)
        ]
        # df = df.drop(columns=binary_cols)
        # convert binary columns to 0/1
        for col in binary_cols:
            df[col] = df[col].map({
                'normal': 1,
                'abnormal': 0,
                'present': 1,
                'notpresent': 0,
                'yes': 1,
                'no': 0,
                'good': 1,
                'poor': 0,
                })


        # drop columns with too many missing values
        df = df.drop(columns=[
            'wc', 
            'rc', 
            'pcv', 
            'sod', 
            'pot',
            'bgr',
            'rbc',
            ])
        
        # column name mapping
        mapping ={
            'age': 'age',
            'bp': 'blood pressure',
            'sg': 'specific gravity',
            'al': 'albumin',
            'su': 'sugar',
            'rbc': 'red blood cells',
            'pc': 'pus cell',
            'pcc': 'pus cell clumps',
            'ba': 'bacteria',
            'bgr': 'blood glucose random',
            'bu': 'blood urea',
            'sc': 'serum creatinine',
            'sod': 'sodium',
            'pot': 'potassium',
            'hemo': 'hemoglobin',
            'pcv': 'packed cell volume',
            'wc': 'white blood cell count',
            'rc': 'red blood cell count',
            'htn': 'hypertension',
            'dm': 'diabetes mellitus',
            'cad': 'coronary artery disease',
            'appet': 'appetite',
            'pe': 'pedal edema',
            'ane': 'anemia',
            'class': 'class',
        }
        df = df.rename(columns=mapping)

        # fix target names change from 'ckd\t' to 'ckd' and 'no' to 'nockd'
        df['class'] = df['class'].str.strip()
        df['class'] = df['class'].replace({'no':'notckd'})

        # strip string in all columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()

        # remove rows with missing values
        df = df.replace('?', pd.NA)
        df = df.dropna()
        
        # remove rows with missing target
        df = df[df['class'].notna()]

        data['y'] = df.pop('class').to_numpy()
        # convert to binary labels
        data['y'][data['y']=='ckd'] = 1
        data['y'][data['y']=='notckd'] = 0

        # convert features to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        data['X'] = df.to_numpy()
        data['feature_names'] = df.columns.to_list()
        data['label_names'] = ['Chronic Kidney Disease', 'Not Chronic Kidney Disease']
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', 
                            'datasets', 'chronic_kidney_disease', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        # shuffle the dataset
        return data


if __name__ == "__main__":
    loader = chronic_kidney_disease_loader()
    print(loader.get_info(long=True))
    loader.plot_dataset()
    # loader.plot_train_test_split()