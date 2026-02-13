from __future__ import annotations

import numpy as np

from data_loaders.embeddings.base import AbstractDimReducer
from data_loaders.embeddings.pca import KernelPCADimReducer, PCADimReducer
from data_loaders.embeddings.tsne import TSNEDimReducer
from data_loaders.embeddings.umap import UMAPDimReducer, UMAPSupervisedDimReducer

REDUCER_MAP = {
    'PCA':             lambda X, y, n: PCADimReducer(X, num_dims=n),
    'kernelPCA':       lambda X, y, n: KernelPCADimReducer(X, num_dims=n),
    'TSNE':            lambda X, y, n: TSNEDimReducer(X, num_dims=n),
    'UMAP':            lambda X, y, n: UMAPDimReducer(X, num_dims=n),
    'UMAP_supervised': lambda X, y, n: UMAPSupervisedDimReducer(X, y_train=y, num_dims=n),
}


class DimReducer:
    """Dimensionality reduction wrapper fitted on training data.

    Fits a reduction model on ``X_train`` and exposes a ``transform`` method
    to project any data into the lower-dimensional space.  If the data already
    has ``<= num_dims`` features, no reduction is applied.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix used to fit the reducer.
    y_train : np.ndarray or None, default=None
        Training labels — only used by supervised methods (``'UMAP_supervised'``).
    reducer : str, default='PCA'
        Reduction algorithm:
            - 'PCA
            - kernelPCA
            - TSNE'
            - 'UMAP'
            - 'UMAP_supervised'
    num_dims : int, default=2
        Number of output dimensions.
    """

    def __init__(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray | None = None,
            reducer: str = 'PCA',
            num_dims: int = 2,
    ) -> None:
        self.reducer_name = reducer
        self.num_dims = num_dims
        self.reducer_model: AbstractDimReducer | None = None

        if X_train.shape[1] > self.num_dims:
            if reducer not in REDUCER_MAP:
                raise ValueError(f"Unknown reducer type: {reducer}")
            self.reducer_model = REDUCER_MAP[reducer](X_train, y_train, num_dims)
        else:
            # data is already low-dim, no embedding needed
            self.reducer_name = 'Feature'

        self.feature_names = [f"{self.reducer_name} {i+1}" for i in range(self.num_dims)]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data into the reduced space.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to transform.

        Returns
        -------
        np.ndarray
            Transformed feature matrix with ``num_dims`` columns, or the
            original ``X`` if no reduction was needed.
        """
        if self.reducer_model is None:
            return X
        return self.reducer_model.transform(X)
