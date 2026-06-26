# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for the types of wine classification dataset
'''
from __future__ import annotations

from typing import Any

from sklearn.datasets import load_wine
from data_loaders.utils import binarise_labels
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict



class WineLoader(AbstractLoader):
    """Load the Wine dataset as a binary classification problem.

    The original 3-class Wine dataset (178 samples, 13 chemical features) is
    converted to binary by keeping class 0 as-is and merging classes 1 and 2
    into a single class 1.

    Dataset stats: 178 samples, 13 features.

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
                         dataset_name='Wine',
                         short_description='Wine cultivar classification from chemical analysis — 3 classes, 13 features',
                         **kwargs)

    def load_data(self) -> DataDict:
        '''
        The wine dataset is a classic and very easy multi-class classification
        dataset.

        =================   ==============
        Classes                          3
        Samples per class        [59,71,48]
        Samples total                  178
        Dimensionality                  13
        Features            real, positive
        =================   ==============

        The copy of UCI ML Wine Data Set dataset is downloaded and modified to fit
        standard format from:
        https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data


        wine dataset (0 vs 1,2)
        returns:
            - data: dict containing 'X', 'y'
        '''
        # get dataset
        data_cls = load_wine()
        # convert to binary dataset: class 0 vs classes 1 and 2 merged into 1
        y = binarise_labels(data_cls.target, {0: 0, 1: 1, 2: 1})
        # shuffling is handled by AbstractLoader.get_data_dict()
        data = {'X': data_cls.data, 'y': y}
        # add name and description
        data['feature_names'] = data_cls.feature_names
        data['description'] = data_cls.DESCR
        names = []
        names.append(f"{data_cls.target_names[0]}")
        names.append(f"{data_cls.target_names[1]} and {data_cls.target_names[2]}")
        data['label_names'] = names
        return data
        

if __name__ == "__main__":
    loader = WineLoader()
    # loader.plot_dataset()
    print(loader)
    # loader.plot_train_test_split()
