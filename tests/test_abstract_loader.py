"""
Tests for data_loaders/abstract_loader.py
"""
import pytest
import numpy as np
from data_loaders.loaders.abstract_loader import AbstractLoader


class InvalidLoader(AbstractLoader):
    """Loader that returns invalid data for testing."""

    def __init__(self, return_value, **kwargs):
        self.return_value = return_value
        super().__init__(dataset_name='Invalid', **kwargs)

    def load_data(self):
        return self.return_value


class TestAbstractLoaderBasics:
    """Tests for basic AbstractLoader functionality."""

    def test_get_X_returns_numpy_array(self, mock_loader):
        """get_X should return a numpy array."""
        X = mock_loader.get_X()
        assert isinstance(X, np.ndarray)

    def test_get_y_returns_numpy_array(self, mock_loader):
        """get_y should return a numpy array."""
        y = mock_loader.get_y()
        assert isinstance(y, np.ndarray)

    def test_X_y_same_num_instances(self, mock_loader):
        """X and y should have same number of instances."""
        X = mock_loader.get_X()
        y = mock_loader.get_y()
        assert X.shape[0] == y.shape[0]

    def test_name_property(self, mock_loader):
        """name property should return dataset_name."""
        assert mock_loader.name == 'Mock Dataset'

    def test_name_property_none(self):
        """name property should handle None dataset_name."""
        loader = InvalidLoader(return_value={'X': np.zeros((10, 2)), 'y': np.zeros(10)})
        loader.dataset_name = None
        assert loader.name == 'No dataset name available'

    def test_get_description(self, mock_loader):
        """get_description should return description from data."""
        desc = mock_loader.get_description()
        assert desc == 'Mock dataset for testing'

    def test_get_description_missing(self):
        """get_description should handle missing description."""
        loader = InvalidLoader(return_value={'X': np.zeros((10, 2)), 'y': np.zeros(10)})
        desc = loader.get_description()
        assert desc == 'No description available'

    def test_get_feature_names(self, mock_loader):
        """get_feature_names should return feature names."""
        names = mock_loader.get_feature_names()
        assert isinstance(names, list)
        assert len(names) == 4

    def test_get_label_names(self, mock_loader):
        """get_label_names should return label names."""
        names = mock_loader.get_label_names()
        assert isinstance(names, list)
        assert len(names) == 2


class TestAbstractLoaderValidation:
    """Tests for AbstractLoader data validation."""

    def test_invalid_return_type_raises(self):
        """Non-dict return from load_data should raise ValueError."""
        loader = InvalidLoader(return_value="not a dict")
        with pytest.raises(ValueError, match="needs to return a dict"):
            loader.get_X()

    def test_missing_X_raises(self):
        """Missing 'X' key should raise ValueError."""
        loader = InvalidLoader(return_value={'y': np.zeros(10)})
        with pytest.raises(ValueError, match="needs to return a dict containing 'X' and 'y'"):
            loader.get_X()

    def test_missing_y_raises(self):
        """Missing 'y' key should raise ValueError."""
        loader = InvalidLoader(return_value={'X': np.zeros((10, 2))})
        with pytest.raises(ValueError, match="needs to return a dict containing 'X' and 'y'"):
            loader.get_X()


class TestAbstractLoaderCaching:
    """Tests for AbstractLoader caching behavior."""

    def test_data_loaded_once(self, mock_loader):
        """Data should only be loaded once (cached)."""
        assert not mock_loader.already_loaded

        _ = mock_loader.get_X()
        assert mock_loader.already_loaded

        # Modify internal data to verify caching
        mock_loader.data['X'][0, 0] = 999
        X = mock_loader.get_X()
        assert X[0, 0] == 999  # Should return cached, modified data

    def test_get_data_dict_returns_dict(self, mock_loader):
        """get_data_dict should return the full data dictionary."""
        data = mock_loader.get_data_dict()
        assert isinstance(data, dict)
        assert 'X' in data
        assert 'y' in data
        assert 'description' in data


class TestAbstractLoaderShuffle:
    """Tests for AbstractLoader shuffle functionality."""

    def test_shuffle_true_changes_order(self):
        """shuffle=True should change data order."""
        from tests.conftest import MockLoader

        loader1 = MockLoader(shuffle=False, set_seed=42)
        loader2 = MockLoader(shuffle=True, set_seed=42)

        X1 = loader1.get_X()
        X2 = loader2.get_X()

        # Data should be different due to shuffling
        assert not np.array_equal(X1, X2)

    def test_shuffle_reproducible(self):
        """Same seed should produce same shuffle."""
        from tests.conftest import MockLoader

        loader1 = MockLoader(shuffle=True, set_seed=42)
        loader2 = MockLoader(shuffle=True, set_seed=42)

        X1 = loader1.get_X()
        X2 = loader2.get_X()

        np.testing.assert_array_equal(X1, X2)


class TestAbstractLoaderDownsample:
    """Tests for AbstractLoader percent_of_data functionality."""

    def test_percent_of_data_reduces_size(self):
        """percent_of_data should reduce dataset size."""
        from tests.conftest import MockLoader

        loader = MockLoader(n_samples=100, percent_of_data=50)
        X = loader.get_X()

        # Should be approximately 50% (with tolerance for class preservation)
        assert 40 <= X.shape[0] <= 60


class TestAbstractLoaderTrainTestSplit:
    """Tests for AbstractLoader train/test split functionality."""

    def test_train_test_split_returns_two_dicts(self, mock_loader):
        """get_train_test_split should return two data dicts."""
        train, test = mock_loader.get_train_test_split()

        assert isinstance(train, dict)
        assert isinstance(test, dict)
        assert 'X' in train and 'y' in train
        assert 'X' in test and 'y' in test

    def test_train_test_split_sizes(self, mock_loader):
        """Split sizes should respect train_size parameter."""
        train, test = mock_loader.get_train_test_split()

        total = len(train['y']) + len(test['y'])
        assert total == 100  # Original size

    def test_train_test_split_no_overlap(self, mock_loader):
        """Train and test should have no overlapping instances."""
        # Use a loader where we can track instances
        from tests.conftest import MockLoader
        loader = MockLoader(n_samples=20, shuffle=False)

        train, test = loader.get_train_test_split()

        # Create sets of row tuples for comparison
        train_rows = set(tuple(row) for row in train['X'])
        test_rows = set(tuple(row) for row in test['X'])

        assert len(train_rows & test_rows) == 0

    def test_train_test_split_custom_size(self):
        """Custom train_size should be respected."""
        from tests.conftest import MockLoader
        loader = MockLoader(n_samples=100, train_size=0.8)

        train, test = loader.get_train_test_split()

        # Train should be ~80% (with class-wise tolerance)
        assert 70 <= len(train['y']) <= 90


class TestAbstractLoaderScaling:
    """Tests for AbstractLoader scaling functionality."""

    def test_scale_normalizes_data(self):
        """scale=True should normalize train data to [-1, 1]."""
        from tests.conftest import MockLoader
        loader = MockLoader(n_samples=100, scale=True)

        train, test = loader.get_train_test_split()

        # Train data should be in [-1, 1] range (with floating point tolerance)
        assert train['X'].min() >= -1.0 - 1e-10
        assert train['X'].max() <= 1.0 + 1e-10

    def test_scale_uses_train_fit(self):
        """Scaling should fit on train and transform test."""
        from tests.conftest import MockLoader
        loader = MockLoader(n_samples=100, scale=True)

        train, test = loader.get_train_test_split()

        # Test data might slightly exceed [-1, 1] since scaled based on train
        # Just verify both are transformed (not original range)
        assert train['X'].max() - train['X'].min() <= 2.5
        assert test['X'].max() - test['X'].min() <= 5  # Wider tolerance for test


class TestAbstractLoaderDimReducer:
    """Tests for AbstractLoader dimensionality reduction."""

    def test_dim_reducer_pca(self):
        """dim_reducer='PCA' should reduce dimensions."""
        from tests.conftest import MockLoader
        loader = MockLoader(n_samples=50, n_features=10, dim_reducer='PCA', reduce_to_dim=2)

        train, test = loader.get_train_test_split()

        assert train['X'].shape[1] == 2
        assert test['X'].shape[1] == 2

    def test_dim_reducer_none_preserves_dims(self):
        """dim_reducer=None should preserve dimensions."""
        from tests.conftest import MockLoader
        loader = MockLoader(n_samples=50, n_features=10, dim_reducer=None)

        train, test = loader.get_train_test_split()

        assert train['X'].shape[1] == 10
        assert test['X'].shape[1] == 10


class TestAbstractLoaderPostProcess:
    """Tests for train_post_process and test_post_process functionality."""

    def test_train_post_process_modifies_train_data(self):
        """train_post_process should modify the training data."""
        from tests.conftest import MockLoader

        def double_features(X, y):
            return X * 2, y

        loader = MockLoader(n_samples=50, train_post_process=double_features)
        loader_no_pp = MockLoader(n_samples=50)

        train, _ = loader.get_train_test_split()
        train_orig, _ = loader_no_pp.get_train_test_split()

        np.testing.assert_array_almost_equal(train['X'], train_orig['X'] * 2)

    def test_test_post_process_modifies_test_data(self):
        """test_post_process should modify the test data."""
        from tests.conftest import MockLoader

        def double_features(X, y):
            return X * 2, y

        loader = MockLoader(n_samples=50, test_post_process=double_features)
        loader_no_pp = MockLoader(n_samples=50)

        _, test = loader.get_train_test_split()
        _, test_orig = loader_no_pp.get_train_test_split()

        np.testing.assert_array_almost_equal(test['X'], test_orig['X'] * 2)

    def test_train_post_process_does_not_affect_test(self):
        """train_post_process should not modify test data."""
        from tests.conftest import MockLoader

        def zero_out(X, y):
            return np.zeros_like(X), y

        loader = MockLoader(n_samples=50, train_post_process=zero_out)
        train, test = loader.get_train_test_split()

        assert np.all(train['X'] == 0)
        assert not np.all(test['X'] == 0)

    def test_test_post_process_does_not_affect_train(self):
        """test_post_process should not modify train data."""
        from tests.conftest import MockLoader

        def zero_out(X, y):
            return np.zeros_like(X), y

        loader = MockLoader(n_samples=50, test_post_process=zero_out)
        train, test = loader.get_train_test_split()

        assert not np.all(train['X'] == 0)
        assert np.all(test['X'] == 0)

    def test_both_post_processes_applied(self):
        """Both train and test post-processes should be applied independently."""
        from tests.conftest import MockLoader

        def add_one(X, y):
            return X + 1, y

        def multiply_three(X, y):
            return X * 3, y

        loader = MockLoader(n_samples=50, train_post_process=add_one, test_post_process=multiply_three)
        loader_orig = MockLoader(n_samples=50)

        train, test = loader.get_train_test_split()
        train_orig, test_orig = loader_orig.get_train_test_split()

        np.testing.assert_array_almost_equal(train['X'], train_orig['X'] + 1)
        np.testing.assert_array_almost_equal(test['X'], test_orig['X'] * 3)

    def test_post_process_can_filter_samples(self):
        """Post-process should support filtering (changing number of samples)."""
        from tests.conftest import MockLoader

        def keep_class_0(X, y):
            mask = y == 0
            return X[mask], y[mask]

        loader = MockLoader(n_samples=100, train_post_process=keep_class_0)
        train, _ = loader.get_train_test_split()

        assert np.all(train['y'] == 0)

    def test_post_process_can_modify_labels(self):
        """Post-process should support modifying labels."""
        from tests.conftest import MockLoader

        def flip_labels(X, y):
            return X, 1 - y

        loader = MockLoader(n_samples=50, train_post_process=flip_labels)
        loader_orig = MockLoader(n_samples=50)

        train, _ = loader.get_train_test_split()
        train_orig, _ = loader_orig.get_train_test_split()

        np.testing.assert_array_equal(train['y'], 1 - train_orig['y'])

    def test_post_process_override_in_get_train_test_split(self):
        """Post-process passed to get_train_test_split should override constructor."""
        from tests.conftest import MockLoader

        def add_one(X, y):
            return X + 1, y

        def add_ten(X, y):
            return X + 10, y

        loader = MockLoader(n_samples=50, train_post_process=add_one)
        loader_orig = MockLoader(n_samples=50)

        # Override the constructor post-process
        train, _ = loader.get_train_test_split(train_post_process=add_ten)
        train_orig, _ = loader_orig.get_train_test_split()

        np.testing.assert_array_almost_equal(train['X'], train_orig['X'] + 10)

    def test_no_post_process_by_default(self):
        """No post-processing should occur when post-process is None (default)."""
        from tests.conftest import MockLoader

        loader1 = MockLoader(n_samples=50)
        loader2 = MockLoader(n_samples=50)

        train1, test1 = loader1.get_train_test_split()
        train2, test2 = loader2.get_train_test_split()

        np.testing.assert_array_equal(train1['X'], train2['X'])
        np.testing.assert_array_equal(test1['X'], test2['X'])

    def test_post_process_applied_after_scaling(self):
        """Post-process should be applied after scaling."""
        from tests.conftest import MockLoader

        def add_hundred(X, y):
            return X + 100, y

        loader = MockLoader(n_samples=50, scale=True, train_post_process=add_hundred)
        train, _ = loader.get_train_test_split()

        # Scaled data is in [-1, 1], adding 100 puts it in [99, 101]
        assert train['X'].min() >= 98


class TestAbstractLoaderInfo:
    """Tests for AbstractLoader info/display methods."""

    def test_get_info_returns_string(self, mock_loader):
        """get_info should return a string."""
        info = mock_loader.get_info()
        assert isinstance(info, str)
        assert 'Mock Dataset' in info

    def test_get_info_long_false(self, mock_loader):
        """get_info(long=False) should return shorter output."""
        short_info = mock_loader.get_info(long=False)
        long_info = mock_loader.get_info(long=True)

        assert len(short_info) < len(long_info)

    def test_str_returns_info(self, mock_loader):
        """__str__ should return get_info output."""
        str_output = str(mock_loader)
        info_output = mock_loader.get_info()

        assert str_output == info_output
