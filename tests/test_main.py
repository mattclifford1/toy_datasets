"""
Tests for data_loaders/main.py (registry and get_dataset)
"""
import pytest
import numpy as np
from data_loaders.main import AVAILABLE_DATASETS, get_dataset
from data_loaders.loaders.abstract_loader import AbstractLoader

# Image datasets are slow to download/load and are tested separately.
IMAGE_DATASETS = (
    'MNIST', 'Fashion-MNIST', 'SVHN', 'EuroSAT',
    'CIFAR-10', 'CIFAR-100', 'CIFAR-10N',
    'PneumoniaMNIST', 'BreastMNIST', 'DermaMNIST',
    'BloodMNIST', 'PathMNIST', 'OCTMNIST',
)


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
        if dataset_name in IMAGE_DATASETS:
            pytest.skip("Image dataset is slow to load; tested separately")

        loader = get_dataset(dataset_name)
        X = loader.get_X()
        y = loader.get_y()

        assert isinstance(X, np.ndarray), f"{dataset_name}: X is not ndarray"
        assert isinstance(y, np.ndarray), f"{dataset_name}: y is not ndarray"
        assert X.shape[0] == y.shape[0], f"{dataset_name}: X/y size mismatch"
        assert X.shape[0] >= 10, f"{dataset_name}: too few instances"
        assert len(X.shape) == 2, f"{dataset_name}: X should be 2D"

    @pytest.mark.parametrize("dataset_name", list(AVAILABLE_DATASETS.keys()))
    def test_X_is_float(self, dataset_name):
        """X feature matrix should be a float numpy array."""
        if dataset_name in IMAGE_DATASETS:
            pytest.skip("Image dataset is slow to load; tested separately")

        loader = get_dataset(dataset_name)
        X = loader.get_X()

        assert np.issubdtype(X.dtype, np.floating), \
            f"{dataset_name}: X dtype is {X.dtype}, expected floating"

    @pytest.mark.parametrize("dataset_name", list(AVAILABLE_DATASETS.keys()))
    def test_y_is_binary_int(self, dataset_name):
        """y label array should be integer dtype with only values 0 and 1."""
        if dataset_name in IMAGE_DATASETS:
            pytest.skip("Image dataset is slow to load; tested separately")

        loader = get_dataset(dataset_name)
        y = loader.get_y()

        assert np.issubdtype(y.dtype, np.integer), \
            f"{dataset_name}: y dtype is {y.dtype}, expected integer"
        unique = set(np.unique(y).tolist())
        assert unique.issubset({0, 1}), \
            f"{dataset_name}: y contains non-binary labels {unique}"

    @pytest.mark.parametrize("dataset_name", list(AVAILABLE_DATASETS.keys()))
    def test_loader_train_test_split(self, dataset_name):
        """All loaders should support train/test split."""
        if dataset_name in IMAGE_DATASETS:
            pytest.skip("Image dataset is slow to load; tested separately")

        loader = get_dataset(dataset_name)
        train, test = loader.get_train_test_split()

        assert 'X' in train and 'y' in train
        assert 'X' in test and 'y' in test
        assert train['X'].shape[0] == len(train['y'])
        assert test['X'].shape[0] == len(test['y'])

    @pytest.mark.parametrize("dataset_name", list(AVAILABLE_DATASETS.keys()))
    def test_loader_has_name(self, dataset_name):
        """All loaders should have a name."""
        if dataset_name in IMAGE_DATASETS:
            pytest.skip("Image dataset is slow to load; tested separately")

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


class TestCIFAR:
    """Separate tests for CIFAR datasets (slow to load)."""

    @pytest.mark.slow
    def test_cifar10_loads(self):
        """CIFAR-10 should return X shape (n, 3072) and binary y."""
        loader = get_dataset('CIFAR-10', size=256)
        X = loader.get_X()
        y = loader.get_y()

        assert isinstance(X, np.ndarray)
        assert X.shape[1] == 3072
        assert X.shape[0] == y.shape[0]
        assert set(np.unique(y).tolist()).issubset({0, 1})

    @pytest.mark.slow
    def test_cifar10_multiclass(self):
        """CIFAR-10 multiclass mode should produce 10 distinct classes."""
        loader = get_dataset('CIFAR-10', size=256, binary=False)
        y = loader.get_y()
        assert len(np.unique(y)) == 10

    @pytest.mark.slow
    def test_cifar100_loads(self):
        """CIFAR-100 should return X shape (n, 3072) and binary y."""
        loader = get_dataset('CIFAR-100', size=256)
        X = loader.get_X()
        y = loader.get_y()

        assert isinstance(X, np.ndarray)
        assert X.shape[1] == 3072
        assert X.shape[0] == y.shape[0]
        assert set(np.unique(y).tolist()).issubset({0, 1})

    @pytest.mark.slow
    def test_cifar100_multiclass(self):
        """CIFAR-100 multiclass mode should produce 100 distinct classes."""
        loader = get_dataset('CIFAR-100', size=2500, binary=False)
        y = loader.get_y()
        assert len(np.unique(y)) == 100

    @pytest.mark.slow
    def test_cifar10n_loads(self):
        """CIFAR-10N auto-downloads its human label file and loads."""
        loader = get_dataset('CIFAR-10N', size=256)
        X = loader.get_X()
        y = loader.get_y()

        assert isinstance(X, np.ndarray)
        assert X.shape[1] == 3072
        assert X.shape[0] == y.shape[0]
        assert set(np.unique(y).tolist()).issubset({0, 1})

    @pytest.mark.slow
    def test_cifar10n_multiclass(self):
        """CIFAR-10N multiclass mode should produce 10 distinct classes."""
        loader = get_dataset('CIFAR-10N', size=2500, binary=False)
        y = loader.get_y()
        assert len(np.unique(y)) == 10


class TestExtraImageLoaders:
    """Separate tests for the additional image loaders (slow to download)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("dataset_name,n_features,n_classes", [
        ('Fashion-MNIST', 784, 10),
        ('SVHN', 3072, 10),
        ('EuroSAT', 12288, 10),
    ])
    def test_image_loader(self, dataset_name, n_features, n_classes):
        """General image loaders return flat float features and binary y."""
        loader = get_dataset(dataset_name, size=512)
        X = loader.get_X()
        y = loader.get_y()
        assert X.shape[1] == n_features
        assert X.shape[0] == y.shape[0]
        assert set(np.unique(y).tolist()).issubset({0, 1})

        multiclass = get_dataset(dataset_name, size=512, binary=False)
        assert len(np.unique(multiclass.get_y())) == n_classes


class TestMedMNIST:
    """Separate tests for the MedMNIST loaders (slow to download)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("dataset_name,n_classes", [
        ('PneumoniaMNIST', 2),
        ('BreastMNIST', 2),
        ('DermaMNIST', 7),
        ('BloodMNIST', 8),
        ('PathMNIST', 9),
        ('OCTMNIST', 4),
    ])
    def test_medmnist_loader(self, dataset_name, n_classes):
        """MedMNIST loaders are binary by default and expose native classes."""
        loader = get_dataset(dataset_name)
        X = loader.get_X()
        y = loader.get_y()
        assert X.shape[1] == 28 * 28 * (3 if dataset_name in
                                        ('DermaMNIST', 'BloodMNIST', 'PathMNIST') else 1)
        assert X.shape[0] == y.shape[0]
        assert set(np.unique(y).tolist()).issubset({0, 1})

        multiclass = get_dataset(dataset_name, binary=False)
        assert len(np.unique(multiclass.get_y())) == n_classes
