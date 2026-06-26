"""
Shared pytest fixtures for toy_datasets tests.
"""
import pytest
import numpy as np
from data_loaders.loaders.abstract_loader import AbstractLoader


class MockLoader(AbstractLoader):
    """A minimal mock loader for testing AbstractLoader functionality."""

    def __init__(self, n_samples=100, n_features=4, n_classes=2, **kwargs):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        super().__init__(dataset_name='Mock Dataset', **kwargs)

    def load_data(self):
        np.random.seed(42)
        X = np.random.randn(self.n_samples, self.n_features)
        y = np.random.randint(0, self.n_classes, self.n_samples)
        return {
            'X': X,
            'y': y,
            'description': 'Mock dataset for testing',
            'feature_names': [f'feature_{i}' for i in range(self.n_features)],
            'label_names': [f'class_{i}' for i in range(self.n_classes)],
        }


class MockImageLoader(AbstractLoader):
    """A minimal mock image loader for testing image-preview functionality.

    Generates flat samples that reshape into ``image_shape`` images.  Set
    ``channels_first`` to mimic torchvision-style ``C, H, W`` storage.
    """

    is_image = True

    def __init__(self, image_shape=(8, 8), channels_first=False,
                 n_samples=40, n_classes=2, **kwargs):
        self.image_shape = image_shape
        self.channels_first = channels_first
        self.n_samples = n_samples
        self.n_classes = n_classes
        super().__init__(dataset_name='Mock Image Dataset', **kwargs)

    def load_data(self):
        np.random.seed(0)
        n_features = int(np.prod(self.image_shape))
        X = np.random.rand(self.n_samples, n_features).astype(np.float32)
        y = np.array([i % self.n_classes for i in range(self.n_samples)])
        return {
            'X': X,
            'y': y,
            'label_names': [f'class_{i}' for i in range(self.n_classes)],
        }


@pytest.fixture
def mock_loader():
    """Fixture providing a fresh MockLoader instance."""
    return MockLoader()


@pytest.fixture
def mock_image_loader():
    """Fixture providing a greyscale MockImageLoader instance."""
    return MockImageLoader()


@pytest.fixture
def mock_loader_small():
    """Fixture providing a small MockLoader for quick tests."""
    return MockLoader(n_samples=20, n_features=2, n_classes=2)


@pytest.fixture
def sample_data():
    """Fixture providing sample data dict."""
    np.random.seed(42)
    return {
        'X': np.random.randn(100, 4),
        'y': np.array([0]*50 + [1]*50),
    }


@pytest.fixture
def imbalanced_data():
    """Fixture providing imbalanced sample data."""
    np.random.seed(42)
    return {
        'X': np.random.randn(100, 4),
        'y': np.array([0]*80 + [1]*20),
    }
