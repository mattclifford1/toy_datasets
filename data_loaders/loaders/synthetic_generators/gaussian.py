# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Generate Gaussian data with flexible covariance and mean configurations.
'''
from __future__ import annotations

from typing import Any

import numpy as np
from data_loaders.utils import set_seed
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


class GaussianGenerator(AbstractLoader):
    """
    Generate 2-class multivariate Gaussian data for binary classification.

    Supports automatic covariance matrix generation (spherical, diagonal,
    symmetric, random) or manual specification. Means can be auto-generated
    with configurable separation or manually specified.

    Parameters
    ----------
    num_samples : int or list of int
        Total samples (split equally) or per-class sample counts [class0, class1].
        Default: [200, 200]
    n_features : int
        Number of features/dimensions.
        Default: 2
    means : list of array-like, optional
        Mean vector for each class [mean0, mean1]. If None, auto-generated
        based on class_separation. Shape: (2, n_features)
    class_separation : float
        Distance between class means when auto-generating.
        Default: 5.0
    covs : list of array-like, optional
        Covariance matrix for each class [cov0, cov1]. If None, auto-generated
        based on cov_type. Shape: (2, n_features, n_features)
    cov_type : str
        Type of auto-generated covariance: 'spherical', 'diagonal', 'symmetric', 'random'.
        Default: 'spherical'
    cov_scale : float or list of float
        Scale factor(s) for covariance. Single value applies to both classes,
        or provide [scale0, scale1]. Default: 1.0
    cov1_scaler : float
        Relative scale of class 1's covariance w.r.t. class 0.
        If cov1_scaler=2, then cov1 = 2 × cov0. Default: 1.0
    cov_correlation : float
        Correlation strength for 'symmetric' cov_type (-1 to 1).
        Default: 0.5
    random_cov_range : tuple
        (min, max) range for random covariance elements.
        Default: (0.5, 2.0)
    shuffle : bool
        Whether to shuffle the data. Default: True
    name : str
        Dataset name. Default: 'Gaussian Synthetic'

    Examples
    --------
    # Simple 2-class with spherical covariance
    >>> loader = GaussianGenerator(num_samples=400, class_separation=3.0)

    # Different scales per class
    >>> loader = GaussianGenerator(cov_type='diagonal', cov_scale=[1.0, 2.0])

    # Class 1 has 3x the spread of class 0
    >>> loader = GaussianGenerator(cov1_scaler=3.0)

    # Custom means with correlated features
    >>> loader = GaussianGenerator(
    ...     means=[[0, 0], [5, 0]],
    ...     cov_type='symmetric',
    ...     cov_correlation=0.7
    ... )

    # Fully custom specification
    >>> loader = GaussianGenerator(
    ...     means=[[0, 0], [3, 3]],
    ...     covs=[[[1, 0.5], [0.5, 1]], [[2, -0.3], [-0.3, 1]]]
    ... )
    """

    def __init__(
        self,
        shuffle: bool = True,
        num_samples: int | list[int] = 200,
        n_features: int = 2,
        means: list[list[float]] | None = None,
        class_separation: float = 5.0,
        covs: list[list[list[float]]] | None = None,
        cov_type: str = 'spherical',
        cov_scale: float | list[float] = 1.0,
        cov1_scaler: float = 1.0,
        cov_correlation: float = 0.5,
        random_cov_range: tuple[float, float] = (0.5, 2.0),
        name: str = 'Gaussian Synthetic',
        **kwargs: Any,
    ) -> None:
        # Infer n_features from means if provided
        if means is not None:
            if len(means) != 2:
                raise ValueError(f"means must have exactly 2 classes, got {len(means)}")
            n_features = len(means[0])

        self.n_features = n_features

        # Handle num_samples
        if isinstance(num_samples, int):
            self.num_samples = [num_samples // 2, num_samples // 2]
        elif isinstance(num_samples, (list, tuple)):
            if len(num_samples) != 2:
                raise ValueError(f"num_samples must have 2 values or int, got {len(num_samples)}")
            self.num_samples = list(num_samples)
        else:
            raise ValueError(f"num_samples must be int or list of 2 ints, got {type(num_samples)}")

        # Generate or use provided means
        if means is not None:
            self.means = [np.array(m) for m in means]
        else:
            self.means = self._generate_means(n_features, class_separation)

        # Validate covs if provided
        if covs is not None and len(covs) != 2:
            raise ValueError(f"covs must have exactly 2 matrices, got {len(covs)}")

        # Generate or use provided covariances
        if covs is not None:
            self.covs = [np.array(c) for c in covs]
        else:
            self.covs = self._generate_covs(
                n_features, cov_type, cov_scale, cov1_scaler,
                cov_correlation, random_cov_range
            )

        # Store config for metadata
        self.cov_type = cov_type if covs is None else 'custom'
        self.cov1_scaler = cov1_scaler
        self.class_separation = class_separation

        super().__init__(
            shuffle=shuffle,
            dataset_name=name,
            short_description='Multi-class overlapping Gaussian distributions',
            **kwargs
        )

    def _generate_means(self, n_features: int, separation: float) -> list[np.ndarray]:
        """Generate class means at origin and diagonal offset."""
        return [
            np.zeros(n_features),
            np.ones(n_features) * separation
        ]

    def _generate_covs(
            self,
            n_features: int,
            cov_type: str,
            scale: float | list[float],
            cov1_scaler: float,
            correlation: float,
            random_range: tuple[float, float],
    ) -> list[np.ndarray]:
        """Generate covariance matrices based on type."""
        # Handle per-class scales
        if isinstance(scale, (int, float)):
            scales = [scale, scale]
        else:
            scales = list(scale)

        covs = []
        for i in range(2):
            cov = self._make_cov_matrix(
                n_features, cov_type, scales[i], correlation, random_range, seed=i
            )
            covs.append(cov)

        # Apply relative scaling: cov1 = cov1_scaler * cov0
        if cov1_scaler != 1.0:
            covs[1] = cov1_scaler * covs[0]

        return covs

    def _make_cov_matrix(
            self,
            n_features: int,
            cov_type: str,
            scale: float,
            correlation: float,
            random_range: tuple[float, float],
            seed: int | None = None,
    ) -> np.ndarray:
        """Create a single covariance matrix of the specified type."""
        if cov_type == 'spherical':
            # Identity scaled - same variance in all directions
            return np.eye(n_features) * scale

        elif cov_type == 'diagonal':
            # Different variance per feature
            if seed is not None:
                np.random.seed(42 + seed)
            diag = np.random.uniform(0.5, 2.0, n_features) * scale
            return np.diag(diag)

        elif cov_type == 'symmetric':
            # Symmetric with specified correlation
            cov = np.eye(n_features) * scale
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    cov[i, j] = correlation * scale
                    cov[j, i] = correlation * scale
            return cov

        elif cov_type == 'random':
            # Random positive definite matrix
            if seed is not None:
                np.random.seed(42 + seed)
            # Generate random matrix and make it positive definite
            A = np.random.uniform(random_range[0], random_range[1],
                                  (n_features, n_features))
            cov = np.dot(A, A.T) / n_features * scale
            return cov

        else:
            raise ValueError(
                f"Unknown cov_type: {cov_type}. "
                f"Choose from: 'spherical', 'diagonal', 'symmetric', 'random'"
            )

    def load_data(self) -> DataDict:
        """Generate the Gaussian data."""
        X_list = []
        y_list = []

        for label, (mean, cov, n_samples) in enumerate(
            zip(self.means, self.covs, self.num_samples)
        ):
            set_seed(self.set_seed)
            X_class = np.random.multivariate_normal(mean, cov, size=n_samples)
            X_list.append(X_class)
            y_list.append(np.full(n_samples, label))

        X = np.vstack(X_list)
        y = np.hstack(y_list)

        data = {
            'X': X,
            'y': y,
            'description': self._make_description(),
            'feature_names': [f'Feature {i+1}' for i in range(self.n_features)],
            'label_names': ['Class 0', 'Class 1'],
        }
        return data

    def _make_description(self) -> str:
        """Generate a description of the dataset configuration."""
        total = sum(self.num_samples)
        desc = f"Gaussian synthetic dataset for binary classification.\n\n"
        desc += f"Total samples: {total}\n"
        desc += f"Features: {self.n_features}\n"
        desc += f"Covariance type: {self.cov_type}\n\n"

        for i, (mean, cov, n) in enumerate(
            zip(self.means, self.covs, self.num_samples)
        ):
            desc += f"Class {i}: {n} samples\n"
            desc += f"  Mean: {np.round(mean, 2).tolist()}\n"
            desc += f"  Cov diagonal: {np.round(np.diag(cov), 2).tolist()}\n"

        if self.cov1_scaler != 1.0:
            desc += f"Cov1 scaler: {self.cov1_scaler}x (relative to class 0)\n"

        return desc


if __name__ == "__main__":
    # Example: 2-class spherical
    print("2-class spherical:")
    loader = GaussianGenerator(num_samples=400, class_separation=4.0)
    print(loader.get_info(long=False))
    loader.plot_dataset(terminal_plot=True)

    # Example: Different scales per class
    print("\nDifferent scales per class:")
    loader = GaussianGenerator(
        cov_type='diagonal',
        cov_scale=[0.5, 2.0],
        class_separation=6.0
    )
    print(loader.get_info(long=False))
    loader.plot_dataset(terminal_plot=True)

    # Example: Custom means with symmetric covariance
    print("\nSymmetric covariance with correlation:")
    loader = GaussianGenerator(
        means=[[0, 0], [4, 4]],
        cov_type='symmetric',
        cov_correlation=0.8
    )
    print(loader.get_info(long=False))
    loader.plot_dataset(terminal_plot=True)
