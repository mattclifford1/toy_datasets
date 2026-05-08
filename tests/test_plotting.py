"""
Tests for plotting functionality with optional axes parameter.
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from data_loaders.main import get_dataset
from data_loaders.plotting.visualisation import plot_dataset


class TestPlotDatasetAxesParameter:
    """Tests for the ax parameter in plot_dataset function."""

    @pytest.fixture
    def sample_data(self):
        """Create simple sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        return X, y

    @pytest.fixture
    def sample_data_split(self):
        """Create train/test split sample data."""
        np.random.seed(42)
        X_train = np.random.randn(80, 5)
        y_train = np.random.randint(0, 2, 80)
        X_test = np.random.randn(20, 5)
        y_test = np.random.randint(0, 2, 20)
        return X_train, y_train, X_test, y_test


    def test_single_axes_provided(self, sample_data):
        """Test plotting on a provided single axes."""
        X, y = sample_data
        fig, ax = plt.subplots()

        result = plot_dataset(X, y, ax=ax, dim_reducer_method='PCA')

        # Should return None when axes provided
        assert result is None

        # Axes should have been plotted on
        assert len(ax.collections) > 0  # Should have scatter plots
        assert ax.get_xlabel() != ''  # Should have labels

        plt.close('all')

    def test_tuple_of_axes_provided(self, sample_data_split):
        """Test plotting on provided tuple of 2 axes."""
        X_train, y_train, X_test, y_test = sample_data_split
        fig, (ax1, ax2) = plt.subplots(1, 2)

        result = plot_dataset(X_train, y_train, X_test, y_test,
                             ax=(ax1, ax2), dim_reducer_method='PCA')

        # Should return None when axes provided
        assert result is None

        # Both axes should have been plotted on
        assert len(ax1.collections) > 0
        assert len(ax2.collections) > 0
        assert ax1.get_xlabel() != ''
        assert ax2.get_xlabel() != ''

        plt.close('all')

    def test_terminal_plot_with_ax_raises_error(self, sample_data):
        """Test that terminal_plot=True with ax raises ValueError."""
        X, y = sample_data
        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="Cannot use terminal_plot=True with provided ax"):
            plot_dataset(X, y, ax=ax, terminal_plot=True, dim_reducer_method='PCA')

        plt.close('all')

    def test_wrong_ax_type_for_single_dataset(self, sample_data):
        """Test that providing tuple of axes for single dataset raises ValueError."""
        X, y = sample_data
        fig, (ax1, ax2) = plt.subplots(1, 2)

        with pytest.raises(ValueError, match="For single/overlay plot, ax should be a single Axes"):
            plot_dataset(X, y, ax=(ax1, ax2), dim_reducer_method='PCA')

        plt.close('all')

    def test_wrong_ax_type_for_split_dataset(self, sample_data_split):
        """Test that providing single axes for train/test split raises ValueError."""
        X_train, y_train, X_test, y_test = sample_data_split
        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="For train/test split, ax should be a tuple/list"):
            plot_dataset(X_train, y_train, X_test, y_test, ax=ax, dim_reducer_method='PCA')

        plt.close('all')

    def test_wrong_number_of_axes_in_tuple(self, sample_data_split):
        """Test that providing wrong number of axes in tuple raises ValueError."""
        X_train, y_train, X_test, y_test = sample_data_split
        fig, axes = plt.subplots(1, 3)

        with pytest.raises(ValueError, match="For train/test split, ax should be a tuple/list of 2 Axes"):
            plot_dataset(X_train, y_train, X_test, y_test, ax=axes, dim_reducer_method='PCA')

        plt.close('all')

    def test_no_show_called_when_axes_provided(self, sample_data, monkeypatch):
        """Test that plt.show() is not called when axes are provided."""
        X, y = sample_data
        fig, ax = plt.subplots()

        show_called = []

        def mock_show():
            show_called.append(True)

        monkeypatch.setattr(plt, 'show', mock_show)

        plot_dataset(X, y, ax=ax, dim_reducer_method='PCA')

        # plt.show() should NOT have been called
        assert len(show_called) == 0

        plt.close('all')

    def test_suptitle_not_set_when_axes_provided(self, sample_data):
        """Test that suptitle is not set when axes are provided."""
        X, y = sample_data
        fig, ax = plt.subplots()

        plot_dataset(X, y, ax=ax, dataset_name="Test Dataset", dim_reducer_method='PCA')

        # Figure should not have a suptitle (we didn't create it)
        # Note: fig._suptitle is None when no suptitle is set
        assert fig._suptitle is None

        plt.close('all')


class TestAbstractLoaderPlotting:
    """Tests for AbstractLoader plotting methods with ax parameter."""

    def test_plot_dataset_with_ax(self):
        """Test that loader.plot_dataset(ax=ax) plots on provided axes."""
        loader = get_dataset('XOR')
        fig, ax = plt.subplots()

        result = loader.plot_dataset(ax=ax)

        # Should return None
        assert result is None

        # Axes should have plots
        assert len(ax.collections) > 0

        plt.close('all')

    def test_plot_train_test_split_with_axes(self):
        """Test that loader.plot_train_test_split(ax=(ax1, ax2)) plots on provided axes."""
        loader = get_dataset('Circles')
        fig, (ax1, ax2) = plt.subplots(1, 2)

        result = loader.plot_train_test_split(ax=(ax1, ax2))

        # Should return None
        assert result is None

        # Both axes should have plots
        assert len(ax1.collections) > 0
        assert len(ax2.collections) > 0

        plt.close('all')


class TestPlottingGrid:
    """Integration test for creating grid of multiple datasets."""

    def test_multiple_datasets_in_grid(self):
        """Test creating a grid of multiple datasets on custom axes."""
        plt.close('all')

        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Plot 4 different datasets
        datasets = ['XOR', 'Moons', 'Circles', 'Blobs']

        for ax, dataset_name in zip(axes.flat, datasets):
            loader = get_dataset(dataset_name)
            result = loader.plot_dataset(ax=ax)
            # Should not create new figure
            assert result is None
            # Axes should have content
            assert len(ax.collections) > 0

        plt.close('all')

    def test_train_test_grid(self):
        """Test creating train/test comparison on custom layout."""
        plt.close('all')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        loader = get_dataset('Gaussian')
        result = loader.plot_train_test_split(ax=(ax1, ax2))

        assert result is None
        assert len(ax1.collections) > 0
        assert len(ax2.collections) > 0

        plt.close('all')
