# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Wheat seed type predcition dataset (3 classes)
UCI dataset: https://archive.ics.uci.edu/ml/datasets/seeds#
instances: 210
attributes: 7
'''
from __future__ import annotations

from typing import Any

import pandas as pd
from data_loaders.utils import binarise_labels
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


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
                         short_description='Geometric grain measurements for wheat variety classification — 3 classes',
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
        df = pd.read_csv(self.local_dataset_path('wheat_seeds'), header=None)
        if drop_class_3:
            df.drop(df[df[7] == 3].index, inplace=True)
            # Kama (1) -> 1, Rosa (2) -> 0
            label_mapping = {1: 1, 2: 0}
            data['label_names'] = ['Rosa', 'Kama']
        if combine_classes:
            # Kama (1) -> 1, Rosa (2) and Canadian (3) -> 0
            label_mapping = {1: 1, 2: 0, 3: 0}
            data['label_names'] = ['Rosa or Canadian', 'Kama']

        data['y'] = binarise_labels(df.pop(7), label_mapping)
        data['X'] = df.to_numpy()
        data['description'] = self.local_dataset_description('wheat_seeds')
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