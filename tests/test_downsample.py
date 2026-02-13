"""
Tests for data_loaders/utils.py
"""
import pytest
import numpy as np
from data_loaders import downsampling
from data_loaders.downsampling import StratifiedSubsampler
from data_loaders.upsampling import AbstractResampler


class TestStratifiedSubsampler:
    """Tests for StratifiedSubsampler class."""

    def test_repr(self):
        """__repr__ should return 'StratifiedSubsample'."""
        assert repr(StratifiedSubsampler(n_samples=10)) == 'StratifiedSubsample'

    def test_is_abstract_resampler(self):
        """StratifiedSubsampler should be an instance of AbstractResampler."""
        sampler = StratifiedSubsampler(n_samples=10)
        assert isinstance(sampler, AbstractResampler)

    def test_call_interface(self):
        """__call__ should return subsampled arrays of the requested size."""
        X = np.random.randn(100, 5)
        y = np.array([0] * 50 + [1] * 50)
        sampler = StratifiedSubsampler(n_samples=20, random_state=42)
        X_sub, y_sub = sampler(X, y)

        assert X_sub.shape == (20, 5)
        assert y_sub.shape == (20,)

    def test_preserves_class_proportions(self):
        """__call__ should approximately preserve class proportions."""
        X = np.random.randn(1000, 5)
        y = np.array([0] * 700 + [1] * 300)
        sampler = StratifiedSubsampler(n_samples=100, random_state=0)
        _, y_sub = sampler(X, y)

        assert 60 <= np.sum(y_sub == 0) <= 80
        assert 20 <= np.sum(y_sub == 1) <= 40


class TestStratifiedSubsample:
    """Tests for stratified_subsample function."""

    def test_stratified_subsample_preserves_ratio(self):
        """Subsampling should approximately preserve class ratios."""
        X = np.random.randn(1000, 5)
        y = np.array([0]*700 + [1]*300)

        X_sub, y_sub = downsampling.stratified_subsample(X, y, n_samples=100)

        # Check approximate ratio (should be ~70:30)
        class_0 = np.sum(y_sub == 0)
        class_1 = np.sum(y_sub == 1)

        assert 60 <= class_0 <= 80
        assert 20 <= class_1 <= 40
        assert len(y_sub) == 100

    def test_stratified_subsample_reproducible(self):
        """Same random_state should produce same subsample."""
        X = np.random.randn(100, 5)
        y = np.array([0]*50 + [1]*50)

        X1, y1 = downsampling.stratified_subsample(X, y, n_samples=20, random_state=42)
        X2, y2 = downsampling.stratified_subsample(X, y, n_samples=20, random_state=42)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)



class TestProportionalDownsample:
    """Tests for proportional_downsample function."""

    def test_proportional_downsample_size(self, sample_data):
        """Downsampling should reduce to approximately percent_of_data."""
        downsampled = downsampling.proportional_downsample(
            sample_data.copy(), percent_of_data=50
        )

        # Should be approximately 50% (with some tolerance)
        assert 40 <= len(downsampled['y']) <= 60

    def test_proportional_downsample_preserves_ratio(self, imbalanced_data):
        """Downsampling should preserve class proportions."""
        original_ratio = np.sum(imbalanced_data['y'] == 0) / len(imbalanced_data['y'])

        downsampled = downsampling.proportional_downsample(
            imbalanced_data.copy(), percent_of_data=50
        )

        new_ratio = np.sum(downsampled['y'] == 0) / len(downsampled['y'])
        # Ratio should be approximately preserved
        assert abs(original_ratio - new_ratio) < 0.15

    def test_proportional_downsample_invalid_percent(self, sample_data):
        """Invalid percent_of_data should raise ValueError."""
        with pytest.raises(ValueError):
            downsampling.proportional_downsample(sample_data.copy(), percent_of_data=0)

        with pytest.raises(ValueError):
            downsampling.proportional_downsample(sample_data.copy(), percent_of_data=101)

    def test_proportional_downsample_minimum_samples(self):
        """Should keep at least 2 samples per class for splits."""
        data = {
            'X': np.random.randn(10, 2),
            'y': np.array([0]*5 + [1]*5)
        }
        downsampled = downsampling.proportional_downsample(data.copy(), percent_of_data=1)

        _, counts = np.unique(downsampled['y'], return_counts=True)
        assert all(c >= 2 for c in counts)
