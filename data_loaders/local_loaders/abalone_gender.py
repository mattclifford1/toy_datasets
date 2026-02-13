# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
abalone gender UCI dataset: https://archive.ics.uci.edu/ml/datasets/Abalone
instances: 4177
'''
from __future__ import annotations

import os
from typing import Any

import pandas as pd
from data_loaders.abstract_loader import AbstractLoader, DataDict


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class abalone_gender_loader(AbstractLoader):
    """Load the UCI Abalone gender dataset.

    Binary classification: predict whether an abalone is Male (class 0) or
    Female (class 1) from physical measurements.  Infant ('I') samples are
    excluded to keep the problem binary.

    Dataset stats: ~3342 samples (after filtering infants), 8 features.
    Source: https://archive.ics.uci.edu/ml/datasets/Abalone

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    minority_reduce_scaler : int, default=10
        Reduce the minority class (Female) in the train set by this factor
        relative to the majority class.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 minority_reduce_scaler: int = 10,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='Abalone Gender',
                         **kwargs)
        
    def load_data(self) -> DataDict:
        """Load and preprocess the Abalone CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                        'datasets', 'abalone', 'data.csv'), header=None)
        df.drop(df[df[0] == 'I'].index, inplace=True)
        df[0] = df[0].map({'M': 0, 'F': 1}).astype('int64')
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
        data['label_names'] = ['Male', 'Female']
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'abalone', 'description.txt'), 'r') as f:
            data['description'] = f.read()

        return data
    

if __name__ == "__main__":
    loader = abalone_gender_loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()
