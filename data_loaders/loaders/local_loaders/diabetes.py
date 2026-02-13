# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for the Pima Indians Diabetes Database: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download
'''
from __future__ import annotations

import os
from typing import Any

import pandas as pd
from data_loaders.abstract_loader import AbstractLoader, DataDict


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class DiabetesPimaIndiansLoader(AbstractLoader):
    """Load the Pima Indians Diabetes Database.

    Binary classification: predict the onset of diabetes (class 1) vs. no
    diabetes (class 0) from diagnostic measurements.

    Dataset stats: 768 samples, 8 features (glucose, BMI, age, etc.).
    Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    minority_reduce_scaler : int, default=10
        Reduce the minority class (Diabetes) in the train set by this factor
        relative to the majority class.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 minority_reduce_scaler: int = 10,
                 **kwargs: Any) -> None:
        # size=0.2, ratio=5   works nicely
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='Diabetes Pima Indians',
                         **kwargs)
        
    def load_data(self) -> DataDict:
        """Load the Pima Indians Diabetes CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..', '..', 
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
        data['label_names'] = ['No Diabetes', 'Diabetes']
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', '..', 'datasets', 'diabetes_pima_indians', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        return data

    
if __name__ == "__main__":
    loader = DiabetesPimaIndiansLoader()
    print(loader.get_info(long=True))
    # loader.plot_dataset()
    loader.plot_train_test_split()