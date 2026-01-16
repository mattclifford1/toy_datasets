# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
loader for the types of wine classification dataset
'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from data_loaders import utils
from data_loaders.abstract_loader import AbstractLoader



class breast_cancer_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.5,
                #  split_ratio=10,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size, 
                        #  split_ratio=split_ratio, 
                         dataset_name='Breast Cancer',
                         **kwargs)
        
    def load_data(self):
        '''
        Load and return the breast cancer Wisconsin dataset (classification).

        The breast cancer dataset is a classic and very easy binary classification
        dataset.

        =================   ==============
        Classes                          2
        Samples per class    212(M),357(B)
        Samples total                  569
        Dimensionality                  30
        Features            real, positive
        =================   ==============

        The copy of UCI ML Breast Cancer Wisconsin (Diagnostic) dataset is
        downloaded from:
        https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
        breast cancer dataset
        returns:
            - data: dict containing 'X', 'y'
        '''
        # size = 0.453 # for even test 
        # get dataset
        data_cls = load_breast_cancer()
        data = {'X': data_cls.data, 'y': data_cls.target}
        # swap labels for minority convention
        data['y'][data['y'] == 1] = 2
        data['y'][data['y'] == 0] = 1
        data['y'][data['y'] == 2] = 0
        # add name and description
        data['feature_names'] = data_cls.feature_names.tolist()
        data['description'] = data_cls.DESCR
        names = []
        names.append(f"{data_cls.target_names[1]}")
        names.append(f"{data_cls.target_names[0]}")
        data['label_names'] = names
        return data
        

if __name__ == "__main__":
    loader = breast_cancer_loader()
    # loader.plot_dataset()
    print(loader)
    # loader.plot_train_test_split()
