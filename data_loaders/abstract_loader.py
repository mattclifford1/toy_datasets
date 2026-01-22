# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Generic class for data loaders to inherit from
'''
from abc import ABC, abstractmethod
import numpy as np
from data_loaders import utils
from data_loaders.visulalisation import plot_dataset


class AbstractLoader(ABC):
    def __init__(self, 
                 shuffle=True,
                 split_size=0.5,
                 split_ratio=None,  # train ratio
                 split_ratio_test=None,  # test ratio: not implemented yet
                 percent_of_data=None,
                 equal_test=False,
                 set_seed=True,
                 dataset_name=None,
                 scale=False,
                 **kwargs):
        self.shuffle = shuffle
        self.split_size = split_size
        self.split_ratio = split_ratio
        self.split_ratio_test = split_ratio_test
        self.percent_of_data = percent_of_data
        self.equal_test = equal_test
        self.already_loaded = False
        self.dataset_name = dataset_name
        self.set_seed = set_seed
        self.scale = scale #TODO: implement scaling in loaders

    
    @abstractmethod
    def load_data(self):
        '''
        returns:
            - data: dict containing 'X', 'y'
        '''
        raise NotImplementedError("This is an abstract class")
    

    def get_train_test_split(self):
        '''
        returns:
            - data: dict containing 'X', 'y'
            - data_test: dict containing 'X', 'y'
        '''
        # split into train, test
        train_data, test_data = utils.proportional_split( 
            self.get_data_dict(), 
            size=self.split_size, 
            ratio=self.split_ratio,
            equal_test=self.equal_test,
            ratio_test=self.split_ratio_test,  # not implemented yet
            seed=self.set_seed
            ) 
        
        # print info
        print(f"\nDataset: {self.get_dataset_name()} - Train/Test split")
        print(f"    - Train instances: {len(train_data['y'])}")
        label, counts = np.unique(train_data['y'], return_counts=True)
        for labels in zip(label, counts):
            print(f"      - Class {labels[0]}: {labels[1]} instances")
        print(f"    - Test instances: {len(test_data['y'])}")
        label, counts = np.unique(test_data['y'], return_counts=True)
        for labels in zip(label, counts):
            print(f"      - Class {labels[0]}: {labels[1]} instances")

        return train_data, test_data


    def get_data_dict(self):
        '''
        call the data loader and shuffle if needed
        returns:
            - data: dict containing 'X', 'y', 'description' (if available)
        '''
        if not self.already_loaded:
            self.already_loaded = True
            self.data = self.load_data()
            # check valid
            if not isinstance(self.data, dict):
                raise ValueError("load_data() needs to return a dict containing 'X' and 'y'")
            if 'X' not in self.data.keys() or 'y' not in self.data.keys():
                raise ValueError("load_data() needs to return a dict containing 'X' and 'y'")
            # shuffle
            if self.shuffle == True:
                self.data = utils.shuffle_data(
                    self.data,
                    seed=self.set_seed
                    ) 
            # downsample if needed
            if self.percent_of_data is not None:
                self.data = utils.proportional_downsample(
                    self.data, 
                    percent_of_data=self.percent_of_data,
                    seed=self.set_seed
                    )
            
        return self.data
    

    def get_X(self):
        data = self.get_data_dict()
        return data['X']
    

    def get_y(self):
        data = self.get_data_dict()
        return data['y']
    

    def get_description(self):
        data = self.get_data_dict()
        return data.get('description', 'No description available')
    
    
    def get_feature_names(self):
        data = self.get_data_dict()
        return data.get('feature_names', 'No feature names available')
    

    def get_label_names(self):
        data = self.get_data_dict()
        return data.get('label_names', [0, 1, 2, 3])
    

    def get_dataset_name(self):
        if hasattr(self, 'dataset_name'):
            if self.dataset_name is not None:
                return self.dataset_name
        return 'No dataset name available'
    
    
    def get_info(self, long=True):
        msg = f"Data Loader for {self.get_dataset_name()}"
        if long == True:
            msg += f"\n\n Description:\n{self.get_description()}"

        feature_name = self.get_feature_names()
        if isinstance(feature_name, list):
            msg += f"\n\n Feature Names:"
            for i, name in enumerate(feature_name):
                msg += f"\n    - Feature {i}: {name}"
        label_names = self.get_label_names()
        if isinstance(label_names, list):
            msg += f"\n\n Label Names:"
            for i, name in enumerate(label_names):
                msg += f"\n    - Label {i}: {name}"

        msg += f"\n\n Dataset Info:"
        msg += f"\n    - Number of features: {self.get_X().shape[1]}"
        msg += f"\n    - Total instances: {len(self.get_y())}"
        label, counts = np.unique(self.get_y(), return_counts=True)
        for i, labels in enumerate(zip(label, counts)):
            if isinstance(label_names, list):
                msg += f"\n      - Class {labels[0]}: {labels[1]} instances ({label_names[i]})"
            else:
                msg += f"\n      - Class {labels[0]}: {labels[1]} instances"
        return msg


    def plot_dataset(self):
        data = self.get_data_dict()
        plot_dataset(
            X=data['X'], 
            y=data['y'],
            X_test=None, 
            y_test=None,
            dataset_name=self.get_dataset_name(),
            label_names=self.get_label_names()
        )


    def plot_train_test_split(self):
        train_data, test_data = self.get_train_test_split()
        plot_dataset(
            X=train_data['X'], 
            y=train_data['y'],
            X_test=test_data['X'], 
            y_test=test_data['y'],
            dataset_name=self.get_dataset_name(),
            label_names=self.get_label_names()
        )

    def __str__(self):
        return self.get_info()
    

    
    
    

