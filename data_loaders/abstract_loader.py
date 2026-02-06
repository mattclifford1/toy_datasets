# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Generic class for data loaders to inherit from
'''
from abc import ABC, abstractmethod
import numpy as np
from data_loaders import utils
from data_loaders.visualisation import plot_dataset
from data_loaders.embeddings import dim_reducer


class AbstractLoader(ABC):
    def __init__(self, 
                 shuffle=True,
                 train_size=0.5,
                 minority_reduce_scaler=None,  # how much to reduce the minority class in the train set (0.5 means reduce to half, 0.1 means reduce to 10% of original size)
                 split_ratio_test=None,  # test ratio: not implemented yet
                 percent_of_data=None,
                 equal_test=False,
                 set_seed=True,
                 dataset_name=None,
                 scale=False,
                 dim_reducer=None,
                 reduce_to_dim=2,
                 **kwargs):
        self.shuffle = shuffle
        self.train_size = train_size
        self.minority_reduce_scaler = minority_reduce_scaler
        self.split_ratio_test = split_ratio_test
        self.percent_of_data = percent_of_data
        self.equal_test = equal_test
        self.already_loaded = False
        self.dataset_name = dataset_name
        self.set_seed = set_seed
        self.scale = scale
        self.dim_reducer = dim_reducer
        self.reduce_to_dim = reduce_to_dim

    
    @abstractmethod
    def load_data(self):
        '''
        returns:
            - data: dict containing 'X', 'y'
        '''
        raise NotImplementedError("This is an abstract class")
    

    def get_train_test_split(self,
                             train_size=None,
                             minority_reduce_scaler=None,
                             equal_test=None,
                             seed=None):
        '''
        returns:
            - data: dict containing 'X', 'y'
            - data_test: dict containing 'X', 'y'
        '''
        # use provided or default
        if train_size is None:
            train_size = self.train_size
        if minority_reduce_scaler is None:
            minority_reduce_scaler = self.minority_reduce_scaler
        if equal_test is None:
            equal_test = self.equal_test
        if seed is None:
            seed = self.set_seed

        # split into train, test
        train_data, test_data = utils.proportional_split( 
            self.get_data_dict(), 
            train_size=train_size, 
            minority_reduce_scaler=minority_reduce_scaler,
            equal_test=equal_test,
            minority_reduce_scaler_test=self.split_ratio_test,  # not implemented yet
            seed=seed
            )
        
        # reduce dims
        if self.dim_reducer is not None:
            reducer = dim_reducer(
                X_train=train_data['X'],
                y_train=train_data['y'],
                reducer=self.dim_reducer,
                num_dims=self.reduce_to_dim
            )
            train_data['X'] = reducer.transform(train_data['X'])
            test_data['X'] = reducer.transform(test_data['X'])
        
        # scale if needed
        if self.scale:
            # only fit the scaler on the train data
            normaliser = utils.normaliser(train_data['X'])
            train_data['X'] = normaliser(train_data['X'])
            test_data['X'] = normaliser(test_data['X'])

        # print info
        print(f"\nDataset: {self.name} - Train/Test split")
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
            if self.shuffle:
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
    
    
    @property
    def name(self):
        if hasattr(self, 'dataset_name'):
            if self.dataset_name is not None:
                return self.dataset_name
        return 'No dataset name available'
    
    def get_info(self, long=True):
        msg = f"Data Loader for {self.name}"
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


    def plot_dataset(self,
                     terminal_plot=False,
                     ax=None):
        """
        Plot the dataset.

        Parameters
        ----------
        terminal_plot : bool, default=False
            If True, render plot in terminal
        ax : matplotlib.axes.Axes, optional
            If provided, plot on this axes instead of creating new figure

        Returns
        -------
        tuple or None
            Returns (fig, ax) when new figure is created, None otherwise
        """
        data = self.get_data_dict()

        return plot_dataset(
            X=data['X'],
            y=data['y'],
            X_test=None,
            y_test=None,
            dataset_name=self.name,
            label_names=self.get_label_names(),
            terminal_plot=terminal_plot,
            ax=ax  # Pass through
        )


    def plot_train_test_split(self,
                              terminal_plot=False,
                              ax=None):
        """
        Create train/test split and plot both datasets.

        Parameters
        ----------
        terminal_plot : bool, default=False
            If True, render plot in terminal
        ax : tuple of 2 matplotlib.axes.Axes, optional
            Tuple of 2 Axes (train_ax, test_ax) to plot on.
            If None (default): create new figure with 2 subplots

        Returns
        -------
        tuple or None
            Returns (fig, [ax1, ax2]) when new figure is created, None otherwise
        """
        train_data, test_data = self.get_train_test_split()

        return plot_dataset(
            X=train_data['X'],
            y=train_data['y'],
            X_test=test_data['X'],
            y_test=test_data['y'],
            dataset_name=self.name,
            label_names=self.get_label_names(),
            terminal_plot=terminal_plot,
            ax=ax  # Pass through
        )

    def __str__(self):
        return self.get_info()
    

class example_dataset(AbstractLoader):
    def __init__(self,
                 **kwargs):
        super().__init__(shuffle=True,
                         dataset_name='Example Dataloader',
                         **kwargs)
    
    def load_data(self):
        '''
        some example data loading function
        '''
        X_sample = np.array([[1, 2, 4],
                             [2, 3, 5],
                             [3, 4, 6],
                             [5, 6, 7],
                             [8, 9, 10],
                             [13, 14, 15]])
        y_sample = np.array([0, 0, 0, 1, 1, 1])
        data = {
            'X': X_sample,
            'y': y_sample,
            'description': 'This is some example data',
            'feature_names': ['feature_1', 'feature_2', 'feature_3'],
            'label_names': ['class_0', 'class_1']
            }
        return data


if __name__ == "__main__":
    # show example dataset how to construct and test the features
    loader = example_dataset(scale=True,
                             dim_reducer='PCA',
                             reduce_to_dim=2)
    print(loader)
    loader.plot_dataset(terminal_plot=True)
    loader.plot_train_test_split(terminal_plot=True)
