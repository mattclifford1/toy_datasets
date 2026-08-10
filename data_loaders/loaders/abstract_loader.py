# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Generic class for data loaders to inherit from
'''
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders import utils
from data_loaders.resampling import downsampling

DataDict = dict[str, Any]

# directory holding packaged dataset files (data_loaders/datasets/)
_DATASETS_DIR = Path(__file__).resolve().parents[1] / 'datasets'


class AbstractLoader(ABC):
    """Base class for all dataset loaders.

    Subclasses must implement :meth:`load_data`, which returns a dict with at
    least ``'X'`` and ``'y'`` keys.  All splitting, shuffling, scaling, and
    dimensionality-reduction logic lives here so individual loaders stay lean.

    Class attributes
    ----------------
    default_dim_reducer : str
        Dimensionality reduction method used by :meth:`plot_dataset` and
        :meth:`plot_train_test_split` when none is specified.  Image loader
        subclasses override this to ``'TSNE'``.

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
    majority_max : int or None, default=None
        If set, cap the majority class of the *train* set at this many
        instances, applied before ``minority_reduce_scaler`` so the requested
        ratio is taken against the capped count. Keeps very large datasets
        trainable while preserving their natural imbalance. The test set is
        untouched.
    train_post_process : callable or None, default=None
        Function ``(X, y) -> (X, y)`` applied to train data after splitting.
    test_post_process : callable or None, default=None
        Function ``(X, y) -> (X, y)`` applied to test data after splitting.
    percent_of_data : float or None, default=None
        If set, downsample the full dataset to this percentage (0-100) while
        preserving class proportions.
    set_seed : bool or int, default=True
        Random seed for shuffling and splitting. True uses 42, False disables.
    dataset_name : str or None, default=None
        Human-readable name used in plot titles and info output.
    short_description : str or None, default=None
        One-line summary shown in :meth:`get_info` output.
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

    default_dim_reducer: str = 'PCA'

    #: Whether this loader's samples are images.  Image loaders set this True
    #: (along with :attr:`image_shape`) to enable example-image previews in
    #: :meth:`plot_class_samples` and the README gallery.
    is_image: bool = False
    #: Shape to reshape a single flattened sample into for display, e.g.
    #: ``(28, 28)`` for greyscale or ``(32, 32, 3)`` for RGB.  ``None`` for
    #: non-image datasets.
    image_shape: tuple[int, ...] | None = None
    #: True when the flattened image stores the channel axis first
    #: (``C, H, W``), as produced by ``torchvision``'s ``ToTensor``.  Such
    #: samples are transposed to ``H, W, C`` for display.
    channels_first: bool = False

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 minority_reduce_scaler: int | None = None,
                 equal_test: bool = False,
                 minority_reduce_scaler_test: int | None = None,
                 majority_max: int | None = None,
                 train_post_process: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
                 test_post_process: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
                 percent_of_data: float | None = None,
                 set_seed: bool | int = True,
                 dataset_name: str | None = None,
                 short_description: str | None = None,
                 scale: bool = False,
                 dim_reducer: str | None = None,
                 reduce_to_dim: int = 2,
                 **kwargs: Any) -> None:
        self.shuffle = shuffle
        self.train_size = train_size
        self.minority_reduce_scaler = minority_reduce_scaler
        self.minority_reduce_scaler_test = minority_reduce_scaler_test
        self.majority_max = majority_max
        self.train_post_process = train_post_process
        self.test_post_process = test_post_process
        self.percent_of_data = percent_of_data
        self.equal_test = equal_test
        self.already_loaded = False
        self.dataset_name = dataset_name
        self.short_description = short_description
        self.set_seed = set_seed
        self.scale = scale
        self.dim_reducer = dim_reducer
        self.reduce_to_dim = reduce_to_dim
        self._load_lock = threading.Lock()
        self.last_dim_reducer: Any = None  # set after plot_dataset / plot_train_test_split


    @abstractmethod
    def load_data(self) -> DataDict:
        '''
        returns:
            - data: dict containing 'X', 'y'
        '''
        raise NotImplementedError("This is an abstract class")


    # ------------------------------------------------------------------
    # helpers for loaders that read files bundled in data_loaders/datasets/
    # ------------------------------------------------------------------
    @staticmethod
    def local_dataset_path(dataset: str, filename: str = 'data.csv') -> str:
        """Absolute path to a file bundled under ``data_loaders/datasets/``.

        Centralises the location of packaged dataset files so individual
        loaders do not each rebuild the same relative path.

        Parameters
        ----------
        dataset : str
            Sub-directory name inside ``data_loaders/datasets/``.
        filename : str, default='data.csv'
            File to locate within that sub-directory.

        Returns
        -------
        str
            Absolute path to the requested file.
        """
        return str(_DATASETS_DIR / dataset / filename)

    @staticmethod
    def local_dataset_description(dataset: str, filename: str = 'description.txt') -> str:
        """Read the description text bundled with a local dataset.

        Parameters
        ----------
        dataset : str
            Sub-directory name inside ``data_loaders/datasets/``.
        filename : str, default='description.txt'
            Description file name within that sub-directory.

        Returns
        -------
        str
            Contents of the description file.
        """
        return (_DATASETS_DIR / dataset / filename).read_text()


    def get_train_test_split(
            self,
            train_size: float | None = None,
            minority_reduce_scaler: int | None = None,
            equal_test: bool | None = None,
            minority_reduce_scaler_test: int | None = None,
            majority_max: int | None = None,
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
            majority_max: if not None, cap the majority class of the train split at this many instances (applied before minority_reduce_scaler)
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
        if majority_max is None:
            majority_max = self.majority_max
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
            majority_max=majority_max,
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


    def get_cross_validation_folds(
            self,
            n_splits: int = 5,
            seed: bool | int | None = None,
            shuffle_folds: bool = True,
            **split_kwargs: Any,
    ) -> tuple[list[tuple[DataDict, DataDict]], DataDict]:
        '''
        split into k stratified cross validation folds, keeping the test set held out

        The train split from :meth:`get_train_test_split` becomes the
        development set that is folded; the test split is returned untouched so
        it stays a clean estimate of generalisation. Class ratios are preserved
        in every fold, which matters on imbalanced data where an unstratified
        fold can contain no minority instances at all.
            n_splits: number of folds, k
            seed: random seed for the split and the folds (defaults to the loader's)
            shuffle_folds: shuffle instances within each class before folding
            split_kwargs: forwarded to :meth:`get_train_test_split` (train_size,
                minority_reduce_scaler, equal_test, ...)

        returns:
            - folds: list of (train_fold, val_fold) data dict pairs, length n_splits
            - test_data: the held-out test data dict
        '''
        if seed is None:
            seed = self.set_seed
        train_data, test_data = self.get_train_test_split(seed=seed, **split_kwargs)
        folds = utils.stratified_kfold_split(
            train_data,
            n_splits=n_splits,
            seed=seed,
            shuffle=shuffle_folds,
        )
        return folds, test_data


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


    def as_image(self, sample: np.ndarray) -> np.ndarray:
        """Reshape one flattened sample into a displayable image array.

        Parameters
        ----------
        sample : np.ndarray
            A single flattened sample (1-D row of ``X``).

        Returns
        -------
        np.ndarray
            Array of shape ``(H, W)`` (greyscale) or ``(H, W, C)`` (colour),
            ready to pass to matplotlib's ``imshow``.

        Raises
        ------
        ValueError
            If this loader is not an image dataset (``is_image`` is False or
            ``image_shape`` is None).
        """
        if not self.is_image or self.image_shape is None:
            raise ValueError(
                f"{self.name} is not an image dataset "
                "(set is_image=True and image_shape on the loader)"
            )
        img = np.asarray(sample).reshape(self.image_shape)
        if self.channels_first and img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        return img

    def sample_images_per_class(
            self,
            n_per_class: int = 4,
            seed: bool | int = 42,
    ) -> dict[int, list[np.ndarray]]:
        """Collect a few example images for each class.

        Parameters
        ----------
        n_per_class : int, default=4
            Number of example images to sample for each class.
        seed : bool or int, default=42
            Seed for the random sample. True uses 42, False disables seeding.

        Returns
        -------
        dict[int, list[np.ndarray]]
            Mapping from class label to a list of displayable image arrays
            (each as returned by :meth:`as_image`).
        """
        data = self.get_data_dict()
        X, y = data['X'], data['y']
        rng_seed = 42 if seed is True else (None if seed is False else seed)
        rng = np.random.default_rng(rng_seed)
        samples: dict[int, list[np.ndarray]] = {}
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            if len(idx) > n_per_class:
                idx = rng.choice(idx, size=n_per_class, replace=False)
            samples[int(cls)] = [self.as_image(X[i]) for i in idx]
        return samples

    def plot_class_samples(
            self,
            n_per_class: int = 4,
            terminal_plot: bool = False,
            axes: Any = None,
    ) -> tuple[Any, Any] | None:
        """Plot a grid of example images for each class.

        Only valid for image loaders (``is_image=True``).

        Parameters
        ----------
        n_per_class : int, default=4
            Number of example images to show per class.
        terminal_plot : bool, default=False
            If True, render the grid in the terminal.
        axes : np.ndarray of matplotlib Axes, optional
            A ``(n_classes, n_per_class)`` array of axes to draw onto. If None,
            a new figure is created.

        Returns
        -------
        tuple or None
            ``(fig, axes)`` when a new figure is created, otherwise None.
        """
        from data_loaders.plotting.visualisation import plot_class_samples
        samples = self.sample_images_per_class(n_per_class=n_per_class)
        return plot_class_samples(
            samples,
            label_names=self.get_label_names(),
            dataset_name=self.name,
            axes=axes,
            terminal_plot=terminal_plot,
        )

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
        msg = self.name
        if self.short_description:
            msg += f" — {self.short_description}"
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


    def fit_dim_reducer(self, method: str | None = None) -> Any:
        """Fit and return a DimReducer on this dataset's data.

        Use this to obtain a fitted reducer that can be passed to
        :meth:`plot_dataset` or :meth:`plot_train_test_split` so that multiple
        plots share the same projection (important for TSNE/UMAP comparisons).

        Parameters
        ----------
        method : str or None, default=None
            Reduction algorithm. If None uses ``self.default_dim_reducer``.
            Supported: ``'PCA'``, ``'kernelPCA'``, ``'TSNE'``, ``'UMAP'``,
            ``'UMAP_supervised'``.

        Returns
        -------
        DimReducer
            Fitted reducer ready to pass to :meth:`plot_dataset` or
            :meth:`plot_train_test_split` via the ``dim_reducer`` parameter.
        """
        from data_loaders.embeddings import DimReducer
        data = self.get_data_dict()
        return DimReducer(data['X'], data['y'], reducer=method or self.default_dim_reducer)

    def plot_dataset(
            self,
            data_override: DataDict | None = None,
            terminal_plot: bool = False,
            ax: Any = None,
            dim_reducer_method: str | None = None,
            dim_reducer: Any = None,
            clf: Callable | None = None,
            y_pred: np.ndarray | None = None,
            show_legend: bool = True,
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
        dim_reducer_method : str or None, default=None
            Dimensionality reduction method for plotting. If None, uses
            ``self.default_dim_reducer``. Ignored when ``dim_reducer`` is given.
        dim_reducer : DimReducer or None, default=None
            Pre-fitted :class:`~data_loaders.embeddings.DimReducer` instance.
            When provided, it is used directly (bypasses ``dim_reducer_method``).
            After the call the reducer is also stored as ``self.last_dim_reducer``.
            Use :meth:`fit_dim_reducer` or a previous ``self.last_dim_reducer``
            to keep the same projection across multiple plots.
        clf : callable, optional
            Classifier callable in original feature space: ``clf(X) -> labels``.
            Misclassified points are marked with 'X' markers. Decision boundary
            regions are always drawn: directly for 2D data, or via a KNN proxy
            fitted in the projected space when dim reduction is applied.
        y_pred : np.ndarray, optional
            Pre-computed predictions for X (used instead of calling clf for point coloring).
        show_legend : bool, default=True
            If False, suppress the legend.

        Returns
        -------
        tuple or None
            Returns (fig, ax) when new figure is created, None otherwise
        """
        from data_loaders.embeddings import DimReducer
        from data_loaders.plotting.visualisation import plot_dataset
        if data_override is not None:
            data = data_override
        else:
            data = self.get_data_dict()

        if dim_reducer is None:
            dim_reducer = DimReducer(data['X'], data['y'],
                                     reducer=dim_reducer_method or self.default_dim_reducer)
        self.last_dim_reducer = dim_reducer

        return plot_dataset(
            X=data['X'],
            y=data['y'],
            X_test=None,
            y_test=None,
            dataset_name=self.name,
            label_names=self.get_label_names(),
            terminal_plot=terminal_plot,
            ax=ax,
            dim_reducer=dim_reducer,
            clf=clf,
            y_pred=y_pred,
            show_legend=show_legend,
        )


    def plot_train_test_split(
            self,
            train_data_override: DataDict | None = None,
            test_data_override: DataDict | None = None,
            terminal_plot: bool = False,
            ax: Any = None,
            dim_reducer_method: str | None = None,
            dim_reducer: Any = None,
            clf: Callable | None = None,
            y_pred: np.ndarray | None = None,
            y_pred_test: np.ndarray | None = None,
            overlay_train_test: bool = False,
            test_alpha: float = 0.3,
            show_legend: bool = True,
    ) -> tuple[Any, Any] | None:
        """
        Create train/test split and plot both datasets.

        Parameters
        ----------
        train_data_override: dict containing 'X', 'y' to plot instead of the dataset's own train split (useful for plotting modified versions of the train data, e.g. after dimensionality reduction)
        test_data_override: dict containing 'X', 'y' to plot instead of the dataset's own test split (useful for plotting modified versions of the test data, e.g. after dimensionality reduction)
        terminal_plot : bool, default=False
            If True, render plot in terminal
        ax : tuple of 2 matplotlib.axes.Axes or single Axes, optional
            For side-by-side layout (default): tuple of 2 Axes (train_ax, test_ax).
            For overlay_train_test=True: single Axes.
            If None (default): create new figure automatically.
        dim_reducer_method : str or None, default=None
            Dimensionality reduction method for plotting. If None, uses
            ``self.default_dim_reducer``. Ignored when ``dim_reducer`` is given.
        dim_reducer : DimReducer or None, default=None
            Pre-fitted :class:`~data_loaders.embeddings.DimReducer` instance.
            When provided, it is used directly (bypasses ``dim_reducer_method``).
            The reducer is fitted on the train data if not supplied. After the
            call it is stored as ``self.last_dim_reducer`` so it can be reused.
        clf : callable, optional
            Classifier callable in original feature space: ``clf(X) -> labels``.
            Misclassified points are marked with 'X' markers. Decision boundary
            regions are always drawn: directly for 2D data, or via a KNN proxy
            fitted in the projected space when dim reduction is applied.
        y_pred : np.ndarray, optional
            Pre-computed predictions for training data (used instead of calling clf).
        y_pred_test : np.ndarray, optional
            Pre-computed predictions for test data (used instead of calling clf).
        overlay_train_test : bool, default=False
            If True, plot train and test on one axes. Train uses filled markers;
            test uses open markers at test_alpha opacity.
        test_alpha : float, default=0.3
            Alpha for test data scatter points when overlay_train_test=True.
        show_legend : bool, default=True
            If False, suppress the legend on all axes.

        Returns
        -------
        tuple or None
            Returns (fig, [ax1, ax2]) for side-by-side, (fig, ax) for overlay,
            when new figure is created. Returns None if ax was provided.
        """
        from data_loaders.embeddings import DimReducer
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

        if dim_reducer is None:
            dim_reducer = DimReducer(train_data['X'], train_data['y'],
                                     reducer=dim_reducer_method or self.default_dim_reducer)
        self.last_dim_reducer = dim_reducer

        return plot_dataset(
            X=train_data['X'],
            y=train_data['y'],
            X_test=test_data['X'],
            y_test=test_data['y'],
            dataset_name=self.name,
            label_names=self.get_label_names(),
            terminal_plot=terminal_plot,
            ax=ax,
            dim_reducer=dim_reducer,
            clf=clf,
            y_pred=y_pred,
            y_pred_test=y_pred_test,
            overlay_train_test=overlay_train_test,
            test_alpha=test_alpha,
            show_legend=show_legend,
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
