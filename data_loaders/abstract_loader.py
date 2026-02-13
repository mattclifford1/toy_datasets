# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Generic class for data loaders to inherit from
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from data_loaders import utils

DataDict = dict[str, Any]


class AbstractLoader(ABC):
    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 minority_reduce_scaler: int | None = None,
                 equal_test: bool = False,
                 minority_reduce_scaler_test: int | None = None,
                 train_post_process: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
                 test_post_process: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
                 percent_of_data: float | None = None,
                 set_seed: bool | int = True,
                 dataset_name: str | None = None,
                 scale: bool = False,
                 dim_reducer: str | None = None,
                 reduce_to_dim: int = 2,
                 **kwargs: Any) -> None:
        self.shuffle = shuffle
        self.train_size = train_size
        self.minority_reduce_scaler = minority_reduce_scaler
        self.minority_reduce_scaler_test = minority_reduce_scaler_test
        self.train_post_process = train_post_process
        self.test_post_process = test_post_process
        self.percent_of_data = percent_of_data
        self.equal_test = equal_test
        self.already_loaded = False
        self.dataset_name = dataset_name
        self.set_seed = set_seed
        self.scale = scale
        self.dim_reducer = dim_reducer
        self.reduce_to_dim = reduce_to_dim


    @abstractmethod
    def load_data(self) -> DataDict:
        '''
        returns:
            - data: dict containing 'X', 'y'
        '''
        raise NotImplementedError("This is an abstract class")


    def get_train_test_split(
            self,
            train_size: float | None = None,
            minority_reduce_scaler: int | None = None,
            equal_test: bool | None = None,
            minority_reduce_scaler_test: int | None = None,
            train_post_process: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
            test_post_process: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
            seed: bool | int | None = None,
            _print_info: bool = False,
    ) -> tuple[DataDict, DataDict]:
        '''
        create a train, test split that preserves the class distributions
        Minority class is assumed to be class 1, majority class is assumed to be class
            train_size: size of the train set (0.5 means equal train, test size)
            minority_reduce_scaler: if not None, scale down the minority class by this factor
            equal_test: if True, balance test set classes first (reduces majority to match minority count)
            minority_reduce_scaler_test: if not None, scale down minority class in test set (applied after equal_test if both set)
            train_post_process: function to apply to the train data after splitting (takes in X_train, y_train and returns modified X_train, y_train)
            test_post_process: function to apply to the test data after splitting (takes in X_test, y_test and returns modified X_test, y_test)
            seed: random seed for reproducibility (True means use default seed, False means do not set seed, int means use that as the seed)
            _print_info: whether to print the class distributions in the train and test sets after splitting
        returns:
            - data: dict containing 'X', 'y'
            - data_test: dict containing 'X', 'y'
        '''
        # use provided or default
        if train_size is None:
            train_size = self.train_size
        if minority_reduce_scaler is None:
            minority_reduce_scaler = self.minority_reduce_scaler
        if minority_reduce_scaler_test is None:
            minority_reduce_scaler_test = self.minority_reduce_scaler_test
        if equal_test is None:
            equal_test = self.equal_test
        if train_post_process is None:
            train_post_process = self.train_post_process
        if test_post_process is None:
            test_post_process = self.test_post_process

        if seed is None:
            seed = self.set_seed

        # split into train, test
        train_data, test_data = utils.proportional_split(
            self.get_data_dict(),
            train_size=train_size,
            minority_reduce_scaler=minority_reduce_scaler,
            equal_test=equal_test,
            minority_reduce_scaler_test=minority_reduce_scaler_test,
            seed=seed
            )

        # reduce dims
        if self.dim_reducer is not None:
            from data_loaders.embeddings import dim_reducer

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
        if _print_info:
            print(f"\nDataset: {self.name} - Train/Test split")
            print(f"    - Train instances: {len(train_data['y'])}")
            label, counts = np.unique(train_data['y'], return_counts=True)
            for labels in zip(label, counts):
                print(f"      - Class {labels[0]}: {labels[1]} instances")
            print(f"    - Test instances: {len(test_data['y'])}")
            label, counts = np.unique(test_data['y'], return_counts=True)
            for labels in zip(label, counts):
                print(f"      - Class {labels[0]}: {labels[1]} instances")

        # post process if needed
        if train_post_process is not None:
            train_data['X'], train_data['y'] = train_post_process(train_data['X'], train_data['y'])
        if test_post_process is not None:
            test_data['X'], test_data['y'] = test_post_process(test_data['X'], test_data['y'])

        return train_data, test_data


    def get_data_dict(self) -> DataDict:
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


    def get_X(self) -> np.ndarray:
        data = self.get_data_dict()
        return data['X']


    def get_y(self) -> np.ndarray:
        data = self.get_data_dict()
        return data['y']


    def get_description(self) -> str:
        data = self.get_data_dict()
        return data.get('description', 'No description available')


    def get_feature_names(self) -> list[str] | str:
        data = self.get_data_dict()
        return data.get('feature_names', 'No feature names available')


    def get_label_names(self) -> list[str | int]:
        data = self.get_data_dict()
        return data.get('label_names', [0, 1, 2, 3])


    @property
    def name(self) -> str:
        if hasattr(self, 'dataset_name'):
            if self.dataset_name is not None:
                return self.dataset_name
        return 'No dataset name available'

    def get_info(self, long: bool = True) -> str:
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


    def plot_dataset(
            self,
            data_override: DataDict | None = None,
            terminal_plot: bool = False,
            ax: Any = None,
    ) -> tuple[Any, Any] | None:
        """
        Plot the dataset.

        Parameters
        ----------
        data_override: dict containing 'X', 'y' to plot instead of the dataset's own data (useful for plotting modified versions of the data, e.g. after dimensionality reduction)
        terminal_plot : bool, default=False
            If True, render plot in terminal
        ax : matplotlib.axes.Axes, optional
            If provided, plot on this axes instead of creating new figure

        Returns
        -------
        tuple or None
            Returns (fig, ax) when new figure is created, None otherwise
        """
        from data_loaders.visualisation import plot_dataset
        if data_override is not None:
            data = data_override
        else:
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


    def plot_train_test_split(
            self,
            train_data_override: DataDict | None = None,
            test_data_override: DataDict | None = None,
            terminal_plot: bool = False,
            ax: Any = None,
    ) -> tuple[Any, Any] | None:
        """
        Create train/test split and plot both datasets.

        Parameters
        ----------
        train_data_override: dict containing 'X', 'y' to plot instead of the dataset's own train split (useful for plotting modified versions of the train data, e.g. after dimensionality reduction)
        test_data_override: dict containing 'X', 'y' to plot instead of the dataset's own test split (useful for plotting modified versions of the test data, e.g. after dimensionality reduction)
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
        from data_loaders.visualisation import plot_dataset

        if train_data_override is None or test_data_override is None:
            train_data_original, test_data_original = self.get_train_test_split()

        if train_data_override is not None:
            train_data = train_data_override
        else: train_data = train_data_original
        if test_data_override is not None:
            test_data = test_data_override
        else:
            test_data = test_data_original

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

    def __str__(self) -> str:
        return self.get_info()


class example_dataset(AbstractLoader):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(shuffle=True,
                         dataset_name='Example Dataloader',
                         **kwargs)

    def load_data(self) -> DataDict:
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
