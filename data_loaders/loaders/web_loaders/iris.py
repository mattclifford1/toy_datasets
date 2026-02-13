# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for iris
'''
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import load_iris
from data_loaders import utils
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict



class IrisLoader(AbstractLoader):
    """Load the Iris dataset as a binary classification problem.

    The original 3-class Iris dataset is converted to binary by merging
    *Setosa* (class 0) and *Virginica* (class 2) into a single class 0,
    leaving *Versicolor* as class 1.

    Dataset stats: 150 samples, 4 features (sepal/petal length and width).

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                #  minority_reduce_scaler=10,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                        #  minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='Iris',
                         **kwargs)

    def load_data(self) -> DataDict:
        '''
        The iris dataset is a classic and very easy multi-class classification
        dataset.

        =================   ==============
        Classes                          3
        Samples per class               50
        Samples total                  150
        Dimensionality                   4
        Features            real, positive
        =================   ==============

        Read more in the :ref:`User Guide <iris_dataset>`.

        .. versionchanged:: 0.20
            Fixed two wrong data points according to Fisher's paper.
            The new version is the same as in R, but not as in the UCI
            Machine Learning Repository.

        iris dataset (0,2 vs 1)
        returns:
            - data: dict containing 'X', 'y'
        '''
        # get dataset
        data_cls = load_iris()
        # convert to binary datatset (0 vs 1,2)
        y = data_cls.target
        y[np.where(y == 2)] = 0
        data = {'X': data_cls.data, 'y': y}
        # shuffle the dataset
        data = utils.shuffle_data(data)
        # add the feature names
        data['feature_names'] = ['Sepal length',
                                'Sepal width',
                                'Petal length',
                                'Petal width']

        # add name and description
        data['feature_names'] = data_cls.feature_names
        data['description'] = data_cls.DESCR
        names = []
        names.append(f"{data_cls.target_names[0]} and {data_cls.target_names[2]}")
        names.append(f"{data_cls.target_names[1]}")
        data['label_names'] = names
        return data
        

if __name__ == "__main__":
    loader = IrisLoader()
    print(loader)
    # loader.plot_train_test_split()
