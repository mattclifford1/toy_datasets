"""
Tests for data_loaders/upsampling.py
"""
import pytest
import numpy as np

from data_loaders.resampling import upsampling
from data_loaders.resampling.upsampling import (
    RandomDuplicateMinorityUpsampler,
    SMOTEUpsampler,
)


class TestRandomDuplicateMinorityUpsampler:
    """Tests for RandomDuplicateMinorityUpsampler."""

    def test_balances_classes(self, imbalanced_data):
        """Imbalanced input should produce equal class counts after resampling."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        up = RandomDuplicateMinorityUpsampler(random_state=42)
        X_res, y_res = up(X, y)

        _, counts = np.unique(y_res, return_counts=True)
        assert counts.min() == counts.max()

    def test_majority_class_unchanged(self, imbalanced_data):
        """Majority class count should stay the same."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        majority_count = int(np.bincount(y).max())

        up = RandomDuplicateMinorityUpsampler(random_state=42)
        _, y_res = up(X, y)

        _, counts = np.unique(y_res, return_counts=True)
        assert counts.max() == majority_count

    def test_reproducible(self, imbalanced_data):
        """Same random_state should produce identical results."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        up1 = RandomDuplicateMinorityUpsampler(random_state=7)
        up2 = RandomDuplicateMinorityUpsampler(random_state=7)

        X1, y1 = up1(X, y)
        X2, y2 = up2(X, y)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_returns_correct_types(self, imbalanced_data):
        """__call__ should return numpy arrays."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        up = RandomDuplicateMinorityUpsampler(random_state=42)
        X_res, y_res = up(X, y)

        assert isinstance(X_res, np.ndarray)
        assert isinstance(y_res, np.ndarray)

    def test_repr(self):
        """__repr__ should include factor and seed info."""
        up = RandomDuplicateMinorityUpsampler(factor='equal', random_state=True)
        assert repr(up) == 'RandomDuplicate(factor=equal, seed=True)'

    def test_factor_float(self, imbalanced_data):
        """Float factor should upsample minority class by that multiplier."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        minority_count = int(np.bincount(y).min())
        factor = 2.0
        up = RandomDuplicateMinorityUpsampler(random_state=42, factor=factor)
        _, y_res = up(X, y)
        _, counts = np.unique(y_res, return_counts=True)
        assert counts.min() == int(minority_count * factor)

    def test_invalid_factor_below_one_raises(self):
        """Factor below 1 should raise ValueError."""
        with pytest.raises(ValueError):
            RandomDuplicateMinorityUpsampler(factor=0.5)

    def test_invalid_factor_string_raises(self):
        """Invalid string factor should raise ValueError."""
        with pytest.raises(ValueError):
            RandomDuplicateMinorityUpsampler(factor='invalid')


@pytest.mark.slow
class TestSMOTEUpsampler:
    """Tests for SMOTEUpsampler."""

    def test_balances_classes(self, imbalanced_data):
        """Imbalanced input should produce equal class counts after SMOTE."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        up = SMOTEUpsampler(random_state=42)
        _, y_res = up(X, y)

        _, counts = np.unique(y_res, return_counts=True)
        assert counts.min() == counts.max()

    def test_synthetic_samples_added(self, imbalanced_data):
        """Total sample count should increase after SMOTE."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        up = SMOTEUpsampler(random_state=42)
        X_res, _ = up(X, y)

        assert X_res.shape[0] > X.shape[0]

    def test_reproducible(self, imbalanced_data):
        """Same seed should produce identical SMOTE results."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        up1 = SMOTEUpsampler(random_state=0)
        up2 = SMOTEUpsampler(random_state=0)

        X1, y1 = up1(X, y)
        X2, y2 = up2(X, y)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_repr(self):
        """__repr__ should return 'SMOTE'."""
        assert repr(SMOTEUpsampler()) == 'SMOTE'

