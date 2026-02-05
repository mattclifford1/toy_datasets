"""
Tests for data_loaders/embeddings.py
"""
import pytest
import numpy as np
from data_loaders.embeddings import dim_reducer


class TestDimReducerPCA:
    """Tests for PCA dimensionality reduction."""

    def test_pca_reduces_dimensions(self):
        """PCA should reduce to specified dimensions."""
        X_train = np.random.randn(100, 10)
        reducer = dim_reducer(X_train, reducer='PCA', num_dims=2)

        X_reduced = reducer.transform(X_train)

        assert X_reduced.shape == (100, 2)

    def test_pca_transform_test_data(self):
        """PCA should transform test data consistently."""
        X_train = np.random.randn(100, 10)
        X_test = np.random.randn(20, 10)

        reducer = dim_reducer(X_train, reducer='PCA', num_dims=3)

        X_train_reduced = reducer.transform(X_train)
        X_test_reduced = reducer.transform(X_test)

        assert X_train_reduced.shape == (100, 3)
        assert X_test_reduced.shape == (20, 3)

    def test_pca_feature_names(self):
        """PCA should generate appropriate feature names."""
        X_train = np.random.randn(50, 5)
        reducer = dim_reducer(X_train, reducer='PCA', num_dims=2)

        assert reducer.feature_names == ['PCA 1', 'PCA 2']


class TestDimReducerKernelPCA:
    """Tests for Kernel PCA dimensionality reduction."""

    def test_kernel_pca_reduces_dimensions(self):
        """Kernel PCA should reduce to specified dimensions."""
        X_train = np.random.randn(100, 10)
        reducer = dim_reducer(X_train, reducer='kernelPCA', num_dims=2)

        X_reduced = reducer.transform(X_train)

        assert X_reduced.shape == (100, 2)


class TestDimReducerUMAP:
    """Tests for UMAP dimensionality reduction."""

    def test_umap_reduces_dimensions(self):
        """UMAP should reduce to specified dimensions."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        reducer = dim_reducer(X_train, reducer='UMAP', num_dims=2)

        X_reduced = reducer.transform(X_train)

        assert X_reduced.shape == (100, 2)

    def test_umap_supervised(self):
        """UMAP supervised should use labels."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.array([0]*50 + [1]*50)

        reducer = dim_reducer(
            X_train, y_train=y_train, reducer='UMAP_supervised', num_dims=2
        )

        X_reduced = reducer.transform(X_train)
        assert X_reduced.shape == (100, 2)


class TestDimReducerTSNE:
    """Tests for t-SNE dimensionality reduction."""

    @pytest.mark.slow
    def test_tsne_reduces_dimensions(self):
        """t-SNE should reduce to 2 dimensions."""
        np.random.seed(42)
        X_train = np.random.randn(50, 10)
        reducer = dim_reducer(X_train, reducer='TSNE', num_dims=2)

        X_reduced = reducer.transform(X_train)

        assert X_reduced.shape == (50, 2)


class TestDimReducerLowDimInput:
    """Tests for dim_reducer with already low-dimensional input."""

    def test_low_dim_input_no_reduction(self):
        """Data already at target dims should not be reduced."""
        X_train = np.random.randn(50, 2)
        reducer = dim_reducer(X_train, reducer='PCA', num_dims=2)

        X_transformed = reducer.transform(X_train)

        # Should return original data unchanged
        np.testing.assert_array_equal(X_transformed, X_train)
        assert reducer.reducer_name == 'Feature'

    def test_low_dim_input_feature_names(self):
        """Low-dim input should have 'Feature' names."""
        X_train = np.random.randn(50, 2)
        reducer = dim_reducer(X_train, reducer='PCA', num_dims=2)

        assert reducer.feature_names == ['Feature 1', 'Feature 2']


class TestDimReducerInvalid:
    """Tests for invalid dim_reducer inputs."""

    def test_unknown_reducer_raises(self):
        """Unknown reducer type should raise ValueError."""
        X_train = np.random.randn(50, 10)

        with pytest.raises(ValueError, match="Unknown reducer type"):
            dim_reducer(X_train, reducer='InvalidReducer', num_dims=2)


class TestDimReducerReproducibility:
    """Tests for dim_reducer reproducibility."""

    def test_pca_reproducible(self):
        """PCA should give same results for same input."""
        np.random.seed(42)
        X_train = np.random.randn(50, 10)

        reducer1 = dim_reducer(X_train, reducer='PCA', num_dims=2)
        reducer2 = dim_reducer(X_train, reducer='PCA', num_dims=2)

        X1 = reducer1.transform(X_train)
        X2 = reducer2.transform(X_train)

        np.testing.assert_array_almost_equal(X1, X2)
