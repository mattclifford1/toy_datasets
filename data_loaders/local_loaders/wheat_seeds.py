# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Wheat seed type predcition dataset (3 classes)
UCI dataset: https://archive.ics.uci.edu/ml/datasets/seeds#
instances: 210
attributes: 7
'''

import os
import pandas as pd
from data_loaders.abstract_loader import AbstractLoader


CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


class wheat_seeds_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 train_size=0.7,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Wheat Seeds',
                         **kwargs)
        
    def load_data(self):
        '''
        options to:
            - we remove class 3 and make it a binary problem
            - combine 2 classes into one
        original classes:
            1. Kama
            2. Rosa
            3. Canadian
        '''
        drop_class_3 = False
        combine_classes = True
        if (drop_class_3 and combine_classes):
            raise ValueError("Wheat seeds Cannot both drop class 3 and combine classes")
        if (not drop_class_3) and (not combine_classes):
            raise ValueError("Wheat seeds Must either drop class 3 or combine classes")
        
        data = {}
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
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
        with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'wheat_seeds', 'description.txt'), 'r') as f:
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
    loader = wheat_seeds_loader()
    print(loader.get_info(long=True))
    loader.plot_dataset()
    # loader.plot_train_test_split()