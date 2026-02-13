from __future__ import annotations

import numpy as np

from data_loaders.embeddings.base import AbstractDimReducer


class PCADimReducer(AbstractDimReducer):
    """PCA dimensionality reducer.

    Parameters
    ----------
    X_train : np.ndarray
        Training data used to fit PCA.
    num_dims : int, default=2
        Number of output dimensions.
    """

    def __init__(self, X_train: np.ndarray, num_dims: int = 2) -> None:
        from sklearn.decomposition import PCA

        self.model = PCA(n_components=num_dims, svd_solver='full').fit(X_train)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.transform(X)


class KernelPCADimReducer(AbstractDimReducer):
    """Kernel PCA dimensionality reducer using an RBF kernel.

    Parameters
    ----------
    X_train : np.ndarray
        Training data used to fit Kernel PCA.
    num_dims : int, default=2
        Number of output dimensions.
    """

    def __init__(self, X_train: np.ndarray, num_dims: int = 2) -> None:
        from sklearn.decomposition import KernelPCA

        self.model = KernelPCA(
            n_components=num_dims, kernel='rbf', n_jobs=-1
        ).fit(X_train)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.transform(X)
