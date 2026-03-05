"""
Tests for thread-safe dataset access.

Each loader caches its data in ``self.data`` after the first load.  If
``get_X()`` / ``get_y()`` return direct references to that cached array,
a caller that mutates the result in-place silently corrupts every subsequent
call — including calls from other threads.  These tests verify that returned
arrays are independent copies so in-place mutation by one thread cannot affect
the loader's internal state or other threads.
"""
import threading

import numpy as np
import pytest

from data_loaders.main import get_dataset


# A representative cross-section of fast-loading datasets
THREAD_TEST_DATASETS = [
    'XOR',
    'Moons',
    'Blobs',
    'Circles',
    'Gaussian',
    'Iris',
    'Wine',
    'Breast Cancer',
]


class TestThreadSafety:
    """Verify concurrent dataset access does not corrupt shared internal state."""

    @pytest.mark.parametrize("dataset_name", THREAD_TEST_DATASETS)
    def test_get_X_returns_copy(self, dataset_name: str) -> None:
        """Mutating the array returned by get_X() must not alter internal data."""
        loader = get_dataset(dataset_name)
        X_ref = loader.get_X().copy()

        # In-place mutation of the returned array
        loader.get_X()[:] = 0.0

        np.testing.assert_array_equal(
            loader.get_X(),
            X_ref,
            err_msg=f"{dataset_name}: get_X() returned an alias — in-place mutation corrupted internal data",
        )

    @pytest.mark.parametrize("dataset_name", THREAD_TEST_DATASETS)
    def test_get_y_returns_copy(self, dataset_name: str) -> None:
        """Mutating the array returned by get_y() must not alter internal data."""
        loader = get_dataset(dataset_name)
        y_ref = loader.get_y().copy()

        loader.get_y()[:] = -1

        np.testing.assert_array_equal(
            loader.get_y(),
            y_ref,
            err_msg=f"{dataset_name}: get_y() returned an alias — in-place mutation corrupted internal data",
        )

    @pytest.mark.parametrize("dataset_name", THREAD_TEST_DATASETS)
    def test_concurrent_mutation_does_not_corrupt_data(self, dataset_name: str) -> None:
        """Multiple threads mutating returned arrays must not corrupt each other's view."""
        loader = get_dataset(dataset_name)
        X_ref = loader.get_X().copy()
        y_ref = loader.get_y().copy()

        assertion_errors: list[AssertionError] = []

        def worker() -> None:
            X = loader.get_X()
            y = loader.get_y()
            # Mutate in-place to simulate a careless caller
            X[:] = 0.0
            y[:] = -1
            # The loader's next call must still return the original data
            try:
                np.testing.assert_array_equal(loader.get_X(), X_ref)
                np.testing.assert_array_equal(loader.get_y(), y_ref)
            except AssertionError as exc:
                assertion_errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not assertion_errors, (
            f"{dataset_name}: internal data was corrupted under concurrent access.\n"
            f"{assertion_errors[0]}"
        )

    @pytest.mark.parametrize("dataset_name", THREAD_TEST_DATASETS)
    def test_concurrent_train_test_split_no_errors(self, dataset_name: str) -> None:
        """Calling get_train_test_split() from multiple threads must not raise."""
        loader = get_dataset(dataset_name)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                train, test = loader.get_train_test_split()
                assert train['X'].shape[0] == len(train['y'])
                assert test['X'].shape[0] == len(test['y'])
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, (
            f"{dataset_name}: exception during concurrent get_train_test_split():\n{errors[0]}"
        )
