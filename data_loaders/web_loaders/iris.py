# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for iris
'''
import numpy as np
from sklearn.datasets import load_iris
from data_loaders import utils
from data_loaders.abstract_loader import AbstractLoader



class iris_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.5,
                #  split_ratio=10,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                        #  split_ratio=split_ratio, 
                         dataset_name='Iris',
                         **kwargs)
        
    def load_data(self):
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
    loader = iris_loader()
    print(loader)
    # loader.plot_train_test_split()
