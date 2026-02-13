from typing import Any

import numpy as np

from data_loaders.upsampling import AbstractResampler
from data_loaders.utils import set_seed


class StratifiedSubsampler(AbstractResampler):
    """Subsample data while preserving class proportions.

    Parameters
    ----------
    n_samples : int
        Number of samples to keep.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, n_samples: int, random_state: int = 42) -> None:
        self.n_samples = n_samples
        self.random_state = random_state

    def __repr__(self) -> str:
        return 'StratifiedSubsample'

    def __call__(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Subsample X and y while preserving class proportions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Label array.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Subsampled (X, y) with ``n_samples`` rows.
        """
        from sklearn.model_selection import train_test_split

        X_sub, _, y_sub, _ = train_test_split(
            X, y,
            train_size=self.n_samples,
            stratify=y,
            random_state=self.random_state,
        )
        return X_sub, y_sub


def stratified_subsample(
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample data while preserving class proportions.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Label array.
    n_samples : int
        Number of samples to keep.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Subsampled (X, y) arrays with ``n_samples`` rows.
    """
    return StratifiedSubsampler(n_samples, random_state)(X, y)


def proportional_downsample(
        data: dict[str, Any],
        percent_of_data: float = 1,
        seed: bool | int = True,
        **kwargs: Any,
) -> dict[str, Any]:
    '''
    downsample data whilst keep the represenetaed class proportion distribution
    the same
        data: data dict holder
        percent_of_data: % of the dataset to downsample to
        seed: True, False, or random seed number
    '''
    if percent_of_data <= 0 or percent_of_data > 100:
        raise ValueError(
            f'percent_of_data needs to be between 0 and 100 instead of :{percent_of_data}')
    set_seed(seed)
    # get current class proportions
    classes, counts = np.unique(data['y'], return_counts=True)
    # now downsample
    new_data_counts = (counts*(percent_of_data/100)).astype(np.uint64)
    # make sure we have at least a sample for train/test splits
    new_data_counts[new_data_counts < 2] = 2
    new_inds = []
    for i, cls in enumerate(classes):
        # get all the inds of current class
        cls_inds = np.where(data['y'] == cls)[0]
        # shuffle all the inds to get a random selection
        np.random.shuffle(cls_inds)
        # now store a subsample of class inds
        sub_sample_of_inds = cls_inds[:new_data_counts[i]]
        new_inds.append(list(sub_sample_of_inds))
    # concat all the inds from each class
    new_inds_arr = np.concatenate(new_inds)
    # now only take new_inds from all data arrays
    instances = data['X'].shape[0]
    for key, val in data.items():
        # apply to all numpy arrays that are data rows
        if isinstance(val, np.ndarray) and data[key].shape[0] == instances:
            # apply the shuffle
            data[key] = val[new_inds_arr]
    return data
