"""
Tests for data_loaders/upsampling.py
"""
import pytest
import numpy as np

from data_loaders import upsampling
from data_loaders.upsampling import (
    RandomDuplicateUpsampler,
    SMOTEUpsampler,
    proportional_upsample,
)


class TestRandomDuplicateUpsampler:
    """Tests for RandomDuplicateUpsampler."""

    def test_balances_classes(self, imbalanced_data):
        """Imbalanced input should produce equal class counts after resampling."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        up = RandomDuplicateUpsampler(random_state=42)
        X_res, y_res = up.fit_resample(X, y)

        _, counts = np.unique(y_res, return_counts=True)
        assert counts.min() == counts.max()

    def test_majority_class_unchanged(self, imbalanced_data):
        """Majority class count should stay the same."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        majority_count = int(np.bincount(y).max())

        up = RandomDuplicateUpsampler(random_state=42)
        _, y_res = up.fit_resample(X, y)

        _, counts = np.unique(y_res, return_counts=True)
        assert counts.max() == majority_count

    def test_reproducible(self, imbalanced_data):
        """Same random_state should produce identical results."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        up1 = RandomDuplicateUpsampler(random_state=7)
        up2 = RandomDuplicateUpsampler(random_state=7)

        X1, y1 = up1.fit_resample(X, y)
        X2, y2 = up2.fit_resample(X, y)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_returns_correct_types(self, imbalanced_data):
        """fit_resample should return numpy arrays."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        up = RandomDuplicateUpsampler(random_state=42)
        X_res, y_res = up.fit_resample(X, y)

        assert isinstance(X_res, np.ndarray)
        assert isinstance(y_res, np.ndarray)


@pytest.mark.slow
class TestSMOTEUpsampler:
    """Tests for SMOTEUpsampler."""

    def test_balances_classes(self, imbalanced_data):
        """Imbalanced input should produce equal class counts after SMOTE."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        up = SMOTEUpsampler(random_state=42)
        _, y_res = up.fit_resample(X, y)

        _, counts = np.unique(y_res, return_counts=True)
        assert counts.min() == counts.max()

    def test_synthetic_samples_added(self, imbalanced_data):
        """Total sample count should increase after SMOTE."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        up = SMOTEUpsampler(random_state=42)
        X_res, _ = up.fit_resample(X, y)

        assert X_res.shape[0] > X.shape[0]

    def test_reproducible(self, imbalanced_data):
        """Same seed should produce identical SMOTE results."""
        X, y = imbalanced_data['X'], imbalanced_data['y']
        up1 = SMOTEUpsampler(random_state=0)
        up2 = SMOTEUpsampler(random_state=0)

        X1, y1 = up1.fit_resample(X, y)
        X2, y2 = up2.fit_resample(X, y)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestProportionalUpsample:
    """Tests for proportional_upsample convenience function."""

    def test_proportional_upsample_random_balances(self, imbalanced_data):
        """strategy='random' should produce balanced classes in data dict."""
        result = proportional_upsample(imbalanced_data.copy(), strategy='random', seed=42)

        _, counts = np.unique(result['y'], return_counts=True)
        assert counts.min() == counts.max()

    @pytest.mark.slow
    def test_proportional_upsample_smote_balances(self, imbalanced_data):
        """strategy='smote' should produce balanced classes in data dict."""
        result = proportional_upsample(imbalanced_data.copy(), strategy='smote', seed=42)

        _, counts = np.unique(result['y'], return_counts=True)
        assert counts.min() == counts.max()

    def test_invalid_strategy_raises(self, imbalanced_data):
        """Unknown strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            proportional_upsample(imbalanced_data.copy(), strategy='bogus')

    def test_other_arrays_resampled(self, imbalanced_data):
        """Extra arrays with the same row count should also be resampled."""
        data = dict(imbalanced_data)
        data['weights'] = np.ones(data['X'].shape[0])

        result = proportional_upsample(data.copy(), strategy='random', seed=42)

        assert result['weights'].shape[0] == result['X'].shape[0]

    def test_reproducible_with_seed(self, imbalanced_data):
        """Same seed should yield identical results."""
        r1 = proportional_upsample(imbalanced_data.copy(), strategy='random', seed=99)
        r2 = proportional_upsample(imbalanced_data.copy(), strategy='random', seed=99)

        np.testing.assert_array_equal(r1['X'], r2['X'])
        np.testing.assert_array_equal(r1['y'], r2['y'])
