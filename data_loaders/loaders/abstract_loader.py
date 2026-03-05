# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Generic class for data loaders to inherit from
'''
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from data_loaders import utils
from data_loaders.resampling import downsampling

DataDict = dict[str, Any]


class AbstractLoader(ABC):
    """Base class for all dataset loaders.

    Subclasses must implement :meth:`load_data`, which returns a dict with at
    least ``'X'`` and ``'y'`` keys.  All splitting, shuffling, scaling, and
    dimensionality-reduction logic lives here so individual loaders stay lean.

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    minority_reduce_scaler : int or None, default=None
        If set, reduce the minority class (class 1) in the *train* set by this
        factor relative to the majority class.
    equal_test : bool, default=False
        If True, balance the test set by downsampling the majority class to
        match the minority count before any ``minority_reduce_scaler_test``.
    minority_reduce_scaler_test : int or None, default=None
        If set, further reduce the minority class in the *test* set by this
        factor (applied after ``equal_test`` if both are set).
    train_post_process : callable or None, default=None
        Function ``(X, y) -> (X, y)`` applied to train data after splitting.
    test_post_process : callable or None, default=None
        Function ``(X, y) -> (X, y)`` applied to test data after splitting.
    percent_of_data : float or None, default=None
        If set, downsample the full dataset to this percentage (0–100) while
        preserving class proportions.
    set_seed : bool or int, default=True
        Random seed for shuffling and splitting. True uses 42, False disables.
    dataset_name : str or None, default=None
        Human-readable name used in plot titles and info output.
    scale : bool, default=False
        If True, apply MinMax normalisation (fitted on train) to both splits.
    dim_reducer : str or None, default=None
        Dimensionality reduction method applied before returning splits.
        Supported values: ``'PCA'``, ``'kernelPCA'``, ``'TSNE'``, ``'UMAP'``,
        ``'UMAP_supervised'``.
    reduce_to_dim : int, default=2
        Target number of dimensions when ``dim_reducer`` is set.
    **kwargs
        Additional keyword arguments (ignored, allows flexible subclass init).
    """

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
        self._load_lock = threading.Lock()


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
            from data_loaders.embeddings import DimReducer

            reducer = DimReducer(
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
            normaliser = utils.Normaliser(train_data['X'])
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
        with self._load_lock:
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
                    self.data = downsampling.proportional_downsample(
                        self.data,
                        percent_of_data=self.percent_of_data,
                        seed=self.set_seed
                        )

        return self.data


    def get_X(self) -> np.ndarray:
        """Return the full feature matrix.

        Returns
        -------
        np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        """
        data = self.get_data_dict()
        return data['X'].copy()


    def get_y(self) -> np.ndarray:
        """Return the full label array.

        Returns
        -------
        np.ndarray
            Integer label array of shape ``(n_samples,)``.
        """
        data = self.get_data_dict()
        return data['y'].copy()


    def get_description(self) -> str:
        """Return a human-readable description of the dataset.

        Returns
        -------
        str
            Dataset description, or ``'No description available'`` if none.
        """
        data = self.get_data_dict()
        return data.get('description', 'No description available')


    def get_feature_names(self) -> list[str] | str:
        """Return the names of each feature column.

        Returns
        -------
        list[str] or str
            List of feature name strings, or ``'No feature names available'``
            if the loader does not provide them.
        """
        data = self.get_data_dict()
        return data.get('feature_names', 'No feature names available')


    def get_label_names(self) -> list[str | int]:
        """Return the human-readable names for each class label.

        Returns
        -------
        list[str or int]
            List of label names indexed by class integer, or ``[0, 1, 2, 3]``
            as a fallback if the loader does not provide them.
        """
        data = self.get_data_dict()
        return data.get('label_names', [0, 1, 2, 3])


    @property
    def name(self) -> str:
        """Human-readable dataset name set at construction time.

        Returns
        -------
        str
            Dataset name, or ``'No dataset name available'`` if not set.
        """
        if hasattr(self, 'dataset_name'):
            if self.dataset_name is not None:
                return self.dataset_name
        return 'No dataset name available'

    def get_info(self, long: bool = True) -> str:
        """Return a formatted summary string for the dataset.

        Parameters
        ----------
        long : bool, default=True
            If True, include the full description text.

        Returns
        -------
        str
            Multi-line summary including feature names, label names, instance
            counts per class, and optionally the full description.
        """
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
        from data_loaders.plotting.visualisation import plot_dataset
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
        from data_loaders.plotting.visualisation import plot_dataset

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


class ExampleDataset(AbstractLoader):
    """Minimal example loader demonstrating how to subclass AbstractLoader.

    Generates a small hard-coded 3-feature, 2-class dataset for testing and
    documentation purposes.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(shuffle=True,
                         dataset_name='Example Dataloader',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Return a small hard-coded example dataset.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'description'``,
            ``'feature_names'``, and ``'label_names'``.
        """
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
    loader = ExampleDataset(scale=True,
                             dim_reducer='PCA',
                             reduce_to_dim=2)
    print(loader)
    loader.plot_dataset(terminal_plot=True)
    loader.plot_train_test_split(terminal_plot=True)
