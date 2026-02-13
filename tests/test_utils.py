"""
Tests for data_loaders/utils.py
"""
import pytest
import numpy as np
from data_loaders import utils


class TestNormaliser:
    """Tests for the Normaliser class."""

    def test_Normaliser_output_range(self):
        """Normaliser should scale data to [-1, 1] range."""
        X = np.array([[0, 100], [50, 200], [100, 300]])
        norm = utils.Normaliser(X)
        X_scaled = norm(X)

        assert X_scaled.min() >= -1.0
        assert X_scaled.max() <= 1.0

    def test_Normaliser_preserves_shape(self):
        """Normaliser should preserve array shape."""
        X = np.random.randn(50, 10)
        norm = utils.Normaliser(X)
        X_scaled = norm(X)

        assert X_scaled.shape == X.shape

    def test_Normaliser_transform_instance(self):
        """transform_instance should work on single rows."""
        X = np.array([[0, 0], [10, 10]])
        norm = utils.Normaliser(X)
        instance = np.array([5, 5])
        result = norm.transform_instance(instance)

        assert result.shape == (2,)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0])

    def test_Normaliser_consistent_scaling(self):
        """Same data should produce same scaled values."""
        X = np.random.randn(100, 5)
        norm = utils.Normaliser(X)

        X_scaled1 = norm(X)
        X_scaled2 = norm(X)

        np.testing.assert_array_equal(X_scaled1, X_scaled2)


class TestSetSeed:
    """Tests for set_seed function."""

    def test_set_seed_with_true(self):
        """set_seed(True) should use default RANDOM_STATE."""
        utils.set_seed(True)
        val1 = np.random.rand()
        utils.set_seed(True)
        val2 = np.random.rand()

        assert val1 == val2

    def test_set_seed_with_int(self):
        """set_seed(int) should use that seed."""
        utils.set_seed(123)
        val1 = np.random.rand()
        utils.set_seed(123)
        val2 = np.random.rand()

        assert val1 == val2

    def test_set_seed_with_false(self):
        """set_seed(False) should not set a fixed seed."""
        # This is harder to test definitively, just verify it runs
        utils.set_seed(False)
        _ = np.random.rand()


class TestShuffleData:
    """Tests for shuffle_data function."""

    def test_shuffle_data_preserves_correspondence(self, sample_data):
        """Shuffling should maintain X-y correspondence."""
        # Create data where X[i,0] == y[i] for easy verification
        data = {
            'X': np.array([[0, 1], [1, 2], [2, 3], [3, 4]]),
            'y': np.array([0, 1, 2, 3])
        }
        shuffled = utils.shuffle_data(data.copy(), seed=42)

        # After shuffle, X[:,0] should still equal y
        np.testing.assert_array_equal(shuffled['X'][:, 0], shuffled['y'])

    def test_shuffle_data_preserves_size(self, sample_data):
        """Shuffling should preserve data size."""
        original_size = len(sample_data['y'])
        shuffled = utils.shuffle_data(sample_data.copy())

        assert len(shuffled['y']) == original_size
        assert shuffled['X'].shape == sample_data['X'].shape

    def test_shuffle_data_reproducible(self, sample_data):
        """Same seed should produce same shuffle."""
        data1 = {'X': sample_data['X'].copy(), 'y': sample_data['y'].copy()}
        data2 = {'X': sample_data['X'].copy(), 'y': sample_data['y'].copy()}

        shuffled1 = utils.shuffle_data(data1, seed=42)
        shuffled2 = utils.shuffle_data(data2, seed=42)

        np.testing.assert_array_equal(shuffled1['X'], shuffled2['X'])
        np.testing.assert_array_equal(shuffled1['y'], shuffled2['y'])


class TestShuffleDataset:
    """Tests for shuffle_dataset function."""

    def test_shuffle_dataset_preserves_correspondence(self):
        """shuffle_dataset should maintain array correspondence."""
        data = {
            'X': np.array([[0, 1], [1, 2], [2, 3]]),
            'y': np.array([0, 1, 2]),
            'extra': np.array([10, 20, 30])
        }
        shuffled = utils.shuffle_dataset(data.copy(), seed=42)

        # Check correspondence is maintained
        for i in range(len(shuffled['y'])):
            idx = shuffled['y'][i]
            assert shuffled['X'][i, 0] == idx
            assert shuffled['extra'][i] == (idx + 1) * 10

    def test_shuffle_dataset_ignores_non_arrays(self):
        """shuffle_dataset should ignore non-ndarray items."""
        data = {
            'X': np.array([[1, 2], [3, 4]]),
            'y': np.array([0, 1]),
            'description': 'test string',
            'feature_names': ['a', 'b']
        }
        shuffled = utils.shuffle_dataset(data.copy(), seed=42)

        assert shuffled['description'] == 'test string'
        assert shuffled['feature_names'] == ['a', 'b']


class TestProportionalSplit:
    """Tests for proportional_split function."""

    def test_proportional_split_sizes(self, sample_data):
        """Split should respect train_size parameter."""
        train, test = utils.proportional_split(sample_data.copy(), train_size=0.8)

        total = len(train['y']) + len(test['y'])
        assert total == 100
        # Allow some tolerance due to class-wise splitting
        assert 70 <= len(train['y']) <= 90

    def test_proportional_split_preserves_classes(self, sample_data):
        """Both splits should contain all classes."""
        train, test = utils.proportional_split(sample_data.copy(), train_size=0.5)

        train_classes = set(np.unique(train['y']))
        test_classes = set(np.unique(test['y']))

        assert 0 in train_classes
        assert 1 in train_classes
        assert 0 in test_classes
        assert 1 in test_classes

    def test_proportional_split_invalid_size(self, sample_data):
        """Invalid train_size should raise ValueError."""
        with pytest.raises(ValueError):
            utils.proportional_split(sample_data.copy(), train_size=0)

        with pytest.raises(ValueError):
            utils.proportional_split(sample_data.copy(), train_size=1.5)

    def test_proportional_split_equal_test(self, imbalanced_data):
        """equal_test=True should balance test set classes."""
        train, test = utils.proportional_split(
            imbalanced_data.copy(), train_size=0.5, equal_test=True
        )

        test_classes, test_counts = np.unique(test['y'], return_counts=True)
        # All classes should have same count in test set
        assert len(set(test_counts)) == 1

    def test_proportional_split_with_ratio(self):
        """minority_reduce_scaler parameter should adjust minority class in train."""
        data = {
            'X': np.random.randn(200, 4),
            'y': np.array([0]*100 + [1]*100)
        }
        train, test = utils.proportional_split(data.copy(), train_size=0.5, minority_reduce_scaler=5)

        train_counts = dict(zip(*np.unique(train['y'], return_counts=True)))
        # With minority_reduce_scaler=5, class 1 should have ~1/5 of class 0
        assert train_counts[1] < train_counts[0]

    def test_minority_reduce_scaler_test_reduces_minority(self):
        """minority_reduce_scaler_test should reduce minority class in test set."""
        data = {
            'X': np.random.randn(150, 4),
            'y': np.array([0]*100 + [1]*50)
        }
        _, test = utils.proportional_split(
            data.copy(), train_size=0.5, minority_reduce_scaler_test=2
        )
        test_counts = dict(zip(*np.unique(test['y'], return_counts=True)))
        # Without equal_test, majority stays at ~50, minority halved to ~12-13
        assert test_counts[1] < test_counts[0]

    def test_minority_reduce_scaler_test_alone(self):
        """minority_reduce_scaler_test without equal_test: majority unchanged, minority halved."""
        data = {
            'X': np.random.randn(150, 4),
            'y': np.array([0]*100 + [1]*50)
        }
        _, test_plain = utils.proportional_split(data.copy(), train_size=0.5)
        _, test_reduced = utils.proportional_split(
            data.copy(), train_size=0.5, minority_reduce_scaler_test=2
        )
        plain_counts = dict(zip(*np.unique(test_plain['y'], return_counts=True)))
        reduced_counts = dict(zip(*np.unique(test_reduced['y'], return_counts=True)))

        # Majority class should be unchanged
        assert plain_counts[0] == reduced_counts[0]
        # Minority class should be roughly halved
        assert reduced_counts[1] == max(int(plain_counts[1] / 2), 1)

    def test_equal_test_then_minority_reduce_scaler_test(self):
        """equal_test then minority_reduce_scaler_test: 100:50 → 50:50 → 50:25."""
        data = {
            'X': np.random.randn(150, 4),
            'y': np.array([0]*100 + [1]*50)
        }
        _, test = utils.proportional_split(
            data.copy(), train_size=0.5, equal_test=True, minority_reduce_scaler_test=2
        )
        test_counts = dict(zip(*np.unique(test['y'], return_counts=True)))

        # After equal_test: majority reduced to match minority (~25 each)
        # After minority_reduce_scaler_test=2: minority halved (~12-13)
        assert test_counts[1] < test_counts[0]

    def test_equal_test_then_minority_reduce_not_equal_to_equal_test_only(self):
        """Combining equal_test + minority_reduce_scaler_test differs from equal_test alone."""
        data = {
            'X': np.random.randn(150, 4),
            'y': np.array([0]*100 + [1]*50)
        }
        _, test_equal_only = utils.proportional_split(
            data.copy(), train_size=0.5, equal_test=True
        )
        _, test_combined = utils.proportional_split(
            data.copy(), train_size=0.5, equal_test=True, minority_reduce_scaler_test=2
        )
        equal_counts = dict(zip(*np.unique(test_equal_only['y'], return_counts=True)))
        combined_counts = dict(zip(*np.unique(test_combined['y'], return_counts=True)))

        # equal_test alone: both classes equal
        assert equal_counts[0] == equal_counts[1]
        # combined: minority is further reduced
        assert combined_counts[1] < combined_counts[0]
        assert combined_counts[1] < equal_counts[1]
