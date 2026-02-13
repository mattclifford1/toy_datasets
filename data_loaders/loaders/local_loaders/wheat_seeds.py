# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Wheat seed type predcition dataset (3 classes)
UCI dataset: https://archive.ics.uci.edu/ml/datasets/seeds#
instances: 210
attributes: 7
'''
from __future__ import annotations

import os
from typing import Any

import pandas as pd
from data_loaders.abstract_loader import AbstractLoader, DataDict


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class WheatSeedsLoader(AbstractLoader):
    """Load the UCI Wheat Seeds dataset as a binary classification problem.

    The original 3-class dataset (Kama=1, Rosa=2, Canadian=3) is converted to
    binary by combining Rosa and Canadian into class 0 and keeping Kama as
    class 1.

    Dataset stats: 210 samples, 7 geometric features of wheat kernels.
    Source: https://archive.ics.uci.edu/ml/datasets/seeds

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.7
        Fraction of data used for training in train/test splits.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.7,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Wheat Seeds',
                         **kwargs)
        
    def load_data(self) -> DataDict:
        """Load and preprocess the Wheat Seeds CSV data into a binary problem.

        The 3-class dataset (Kama, Rosa, Canadian) is collapsed to 2 classes
        by combining Rosa (2) and Canadian (3) into class 0, keeping Kama (1)
        as class 1.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        drop_class_3 = False
        combine_classes = True
        if (drop_class_3 and combine_classes):
            raise ValueError("Wheat seeds Cannot both drop class 3 and combine classes")
        if (not drop_class_3) and (not combine_classes):
            raise ValueError("Wheat seeds Must either drop class 3 or combine classes")
        
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..', '..', 
                        'datasets', 'wheat_seeds', 'data.csv'), header=None)
        if drop_class_3:
            df.drop(df[df[7] == 3].index, inplace=True)
            df = df.replace({7: {2: 0}})
            data['label_names'] = ['Rosa', 'Kama']
        if combine_classes:
            df = df.replace({7: {2: 0, 3: 0}})
            data['label_names'] = ['Rosa or Canadian', 'Kama']

        data['y'] = df.pop(7).to_numpy()  # type: ignore
        data['X'] = df.to_numpy()
        # add name and description
        with open(os.path.join(CURRENT_FILE, '..', '..', 'datasets', 'wheat_seeds', 'description.txt'), 'r') as f:
            data['description'] = f.read()
        data['feature_names'] = ['area',
                                'perimeter',
                                'compactness',
                                'length of kernel',
                                'width of kernel',
                                'asymmetry coefficient',
                                'length of kernel groove']
        return data


if __name__ == "__main__":
    loader = WheatSeedsLoader()
    print(loader.get_info(long=True))
    loader.plot_dataset()
    # loader.plot_train_test_split()