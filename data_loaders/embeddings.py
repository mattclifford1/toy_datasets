from __future__ import annotations

from typing import Any

import numpy as np


class dim_reducer:
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray | None = None,
                 reducer: str = 'PCA',
                 num_dims: int = 2) -> None:
        self.reducer_name = reducer
        self.num_dims = num_dims
        if X_train.shape[1] > self.num_dims:
            if reducer == 'TSNE':
                self.reducer_model: Any = TNSE_dim_reducer(X_train)
            elif reducer == 'PCA':
                from sklearn.decomposition import PCA
                self.reducer_model = PCA(n_components=num_dims, svd_solver='full').fit(X_train)
            elif reducer == 'kernelPCA':
                from sklearn.decomposition import KernelPCA
                self.reducer_model = KernelPCA(
                    n_components=num_dims, kernel='rbf', n_jobs=-1).fit(X_train)
            elif reducer == 'UMAP':
                import umap
                self.reducer_model = umap.UMAP(n_components=num_dims).fit(X_train)
            elif reducer == 'UMAP_supervised':
                import umap
                self.reducer_model = umap.UMAP(n_components=num_dims).fit(X_train, y=y_train)
            else:
                raise ValueError(f"Unknown reducer type: {reducer}")
        else:
            # data is already low dim, no embedding needed
            self.reducer_name = 'Feature'
            self.reducer_model = None
        self.feature_names = [f"{self.reducer_name} {i+1}" for i in range(self.num_dims)]


    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reducer_model is None:
            return X  # return original data if no reduction was applied
        return self.reducer_model.transform(X)


class TNSE_dim_reducer:
    def __init__(
            self,
            X_train: np.ndarray,
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
            n_components=2,
            perplexity=self.perplexity,
            random_state=self.random_state,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
        )
        self.embedding_ = self.model.fit(X_train)
        self.transform = self.embedding_.transform

    def get_transform(self, X: np.ndarray) -> np.ndarray:
        # Transform ONLY (no fit) for test/other data
        return self.transform(X)
