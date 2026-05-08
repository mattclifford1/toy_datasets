# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for Wisconsins breast cancer: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
They describe characteristics of the cell nuclei present in the image
A few of the images can be found at http://www.cs.wisc.edu/~street/images/
'''
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import numpy as np
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class BreastCancerWLoader(AbstractLoader):
    """Load the Wisconsin Breast Cancer (Diagnostic) dataset from a local CSV.

    Binary classification: Benign (class 0) vs. Malignant (class 1) based on
    features computed from digitised fine-needle-aspirate images.

    Dataset stats: 569 samples, 30 features.
    Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.45
        Fraction of data used for training in train/test splits.
    minority_reduce_scaler : int, default=10
        Reduce the minority class (Malignant) in the train set by this factor
        relative to the majority class.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.45,
                 minority_reduce_scaler: int = 10,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='Wisconsin Breast Cancer',
                         **kwargs)
        
    def load_data(self) -> DataDict:
        """Load the Wisconsin Breast Cancer CSV data.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..', '..', 
                        'datasets', 'breast_cancer_Wisconsin', 'data.csv'))
        data['y'] = df.pop('diagnosis').map({'B': 0, 'M': 1}).to_numpy()
        df.pop('ID')
        data['X'] = df.to_numpy()
        data['feature_names'] = df.columns.to_list()
        data['label_names'] = ['Benign', 'Malignant']
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', '..', 'datasets', 'breast_cancer_Wisconsin', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        return data


if __name__ == "__main__":
    loader = BreastCancerWLoader()
    print(loader)
    # loader.plot_dataset()
    # loader.plot_train_test_split()