from __future__ import annotations

import numpy as np

from data_loaders.embeddings.base import AbstractDimReducer


class UMAPDimReducer(AbstractDimReducer):
    """UMAP dimensionality reducer.

    Parameters
    ----------
    X_train : np.ndarray
        Training data used to fit UMAP.
    num_dims : int, default=2
        Number of output dimensions.
    """

    def __init__(self, X_train: np.ndarray, num_dims: int = 2) -> None:
        import umap

        self.model = umap.UMAP(n_components=num_dims).fit(X_train)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.transform(X)


class UMAPSupervisedDimReducer(AbstractDimReducer):
    """Supervised UMAP dimensionality reducer.

    Uses label information during fitting to produce class-aware embeddings.

    Parameters
    ----------
    X_train : np.ndarray
        Training data used to fit UMAP.
    y_train : np.ndarray
        Training labels used to guide the embedding.
    num_dims : int, default=2
        Number of output dimensions.
    """

    def __init__(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            num_dims: int = 2,
    ) -> None:
        import umap

        self.model = umap.UMAP(n_components=num_dims).fit(X_train, y=y_train)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.transform(X)
