"""
Tests for data_loaders/main.py (registry and get_dataset)
"""
import pytest
import numpy as np
from data_loaders.main import AVAILABLE_DATASETS, get_dataset
from data_loaders.abstract_loader import AbstractLoader


class TestAvailableDatasets:
    """Tests for the AVAILABLE_DATASETS registry."""

    def test_registry_is_dict(self):
        """AVAILABLE_DATASETS should be a dictionary."""
        assert isinstance(AVAILABLE_DATASETS, dict)

    def test_registry_not_empty(self):
        """Registry should contain datasets."""
        assert len(AVAILABLE_DATASETS) > 0

    def test_registry_values_are_classes(self):
        """All registry values should be loader classes."""
        for name, loader_class in AVAILABLE_DATASETS.items():
            assert callable(loader_class), f"{name} is not callable"

    def test_known_datasets_present(self):
        """Key datasets should be in registry."""
        expected = ['XOR', 'Iris', 'Wine', 'MNIST', 'Moons', 'Blobs']
        for name in expected:
            assert name in AVAILABLE_DATASETS, f"{name} missing from registry"


class TestGetDataset:
    """Tests for the get_dataset function."""

    def test_get_dataset_returns_loader(self):
        """get_dataset should return an AbstractLoader instance."""
        loader = get_dataset('XOR')
        assert isinstance(loader, AbstractLoader)

    def test_get_dataset_invalid_name(self):
        """Invalid dataset name should raise ValueError."""
        with pytest.raises(ValueError, match="not available"):
            get_dataset('NonExistentDataset')

    def test_get_dataset_passes_kwargs(self):
        """get_dataset should pass kwargs to loader."""
        loader = get_dataset('XOR', shuffle=False, train_size=0.7)
        assert loader.shuffle == False
        assert loader.train_size == 0.7

    def test_get_dataset_case_sensitive(self):
        """Dataset names should be case sensitive."""
        # 'XOR' exists but 'xor' should not
        with pytest.raises(ValueError):
            get_dataset('xor')


class TestLoaderIntegration:
    """Integration tests for loaders via get_dataset."""

    @pytest.mark.parametrize("dataset_name", [
        'XOR',
        'Moons',
        'Blobs',
        'Circles',
        'Gaussian',
    ])
    def test_synthetic_loaders(self, dataset_name):
        """Synthetic loaders should return valid data."""
        loader = get_dataset(dataset_name)
        X = loader.get_X()
        y = loader.get_y()

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] >= 10

    @pytest.mark.parametrize("dataset_name", [
        'Iris',
        'Wine',
        'Breast Cancer',
    ])
    def test_sklearn_loaders(self, dataset_name):
        """sklearn-based loaders should return valid data."""
        loader = get_dataset(dataset_name)
        X = loader.get_X()
        y = loader.get_y()

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]

    @pytest.mark.parametrize("dataset_name", [
        'Banknote Authentication',
        'Diabetes Pima Indian',
        'Wheat Seeds',
    ])
    def test_local_loaders(self, dataset_name):
        """Local CSV-based loaders should return valid data."""
        loader = get_dataset(dataset_name)
        X = loader.get_X()
        y = loader.get_y()

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]


class TestAllDatasets:
    """Comprehensive tests for all available datasets."""

    @pytest.mark.parametrize("dataset_name", list(AVAILABLE_DATASETS.keys()))
    def test_loader_returns_valid_X_y(self, dataset_name):
        """All loaders should return valid X and y arrays."""
        if dataset_name == 'MNIST':
            pytest.skip("MNIST is slow to load; tested separately")

        loader = get_dataset(dataset_name)
        X = loader.get_X()
        y = loader.get_y()

        assert isinstance(X, np.ndarray), f"{dataset_name}: X is not ndarray"
        assert isinstance(y, np.ndarray), f"{dataset_name}: y is not ndarray"
        assert X.shape[0] == y.shape[0], f"{dataset_name}: X/y size mismatch"
        assert X.shape[0] >= 10, f"{dataset_name}: too few instances"
        assert len(X.shape) == 2, f"{dataset_name}: X should be 2D"

    @pytest.mark.parametrize("dataset_name", list(AVAILABLE_DATASETS.keys()))
    def test_loader_train_test_split(self, dataset_name):
        """All loaders should support train/test split."""
        if dataset_name == 'MNIST':
            pytest.skip("MNIST is slow to load; tested separately")

        loader = get_dataset(dataset_name)
        train, test = loader.get_train_test_split()

        assert 'X' in train and 'y' in train
        assert 'X' in test and 'y' in test
        assert train['X'].shape[0] == len(train['y'])
        assert test['X'].shape[0] == len(test['y'])

    @pytest.mark.parametrize("dataset_name", list(AVAILABLE_DATASETS.keys()))
    def test_loader_has_name(self, dataset_name):
        """All loaders should have a name."""
        if dataset_name == 'MNIST':
            pytest.skip("MNIST is slow to load; tested separately")

        loader = get_dataset(dataset_name)
        assert loader.name != 'No dataset name available'


class TestMNIST:
    """Separate tests for MNIST (slow to load)."""

    @pytest.mark.slow
    def test_mnist_loads(self):
        """MNIST should load successfully."""
        loader = get_dataset('MNIST')
        X = loader.get_X()
        y = loader.get_y()

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]

    @pytest.mark.slow
    def test_mnist_binary_mode(self):
        """MNIST binary mode should have 2 classes."""
        loader = get_dataset('MNIST', binary=True)
        y = loader.get_y()

        unique_classes = np.unique(y)
        assert len(unique_classes) == 2
