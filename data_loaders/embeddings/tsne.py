from __future__ import annotations

import numpy as np

from data_loaders.embeddings.base import AbstractDimReducer


class TSNEDimReducer(AbstractDimReducer):
    """t-SNE dimensionality reducer using the openTSNE library.

    Fits a t-SNE embedding on ``X_train`` and stores the fitted embedding so
    that new data can be projected via ``transform``.

    Parameters
    ----------
    X_train : np.ndarray
        Training data used to fit the t-SNE model.
    num_dims : int, default=2
        Number of output dimensions.
    perplexity : int, default=30
        t-SNE perplexity (roughly the expected cluster size).
    random_state : int, default=42
        Random seed for reproducibility.
    n_iter : int, default=1000
        Number of optimisation iterations.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all available cores).
    """

    def __init__(
            self,
            X_train: np.ndarray,
            num_dims: int = 2,
            perplexity: int = 30,
            random_state: int = 42,
            n_iter: int = 1000,
            n_jobs: int = -1,
    ) -> None:
        from openTSNE import TSNE as OpenTSNE

        self.perplexity = perplexity
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_jobs = n_jobs

        self.model = OpenTSNE(
            n_components=num_dims,
            perplexity=self.perplexity,
            random_state=self.random_state,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
        )
        self.embedding_ = self.model.fit(X_train)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data using the fitted t-SNE embedding (no re-fitting).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to transform.

        Returns
        -------
        np.ndarray
            Reduced embedding of ``X``.
        """
        return self.embedding_.transform(X)
