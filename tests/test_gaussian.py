"""
Tests for data_loaders/synthetic_generators/gaussian.py
"""
import pytest
import numpy as np
from data_loaders.loaders.synthetic_generators.gaussian import GaussianGenerator


class TestGaussianBasics:
    """Basic functionality tests."""

    def test_default_parameters(self):
        """Default parameters should create valid 2-class dataset."""
        loader = GaussianGenerator()
        X = loader.get_X()
        y = loader.get_y()

        assert X.shape == (200, 2)
        assert y.shape == (200,)
        assert set(np.unique(y)) == {0, 1}

    def test_num_samples_int(self):
        """Integer num_samples should split equally."""
        loader = GaussianGenerator(num_samples=100)
        y = loader.get_y()

        assert len(y) == 100
        assert np.sum(y == 0) == 50
        assert np.sum(y == 1) == 50

    def test_num_samples_list(self):
        """List num_samples should set per-class counts."""
        loader = GaussianGenerator(num_samples=[100, 200])
        y = loader.get_y()

        assert len(y) == 300
        assert np.sum(y == 0) == 100
        assert np.sum(y == 1) == 200

    def test_n_features(self):
        """n_features should set dimensionality."""
        loader = GaussianGenerator(n_features=5)
        X = loader.get_X()

        assert X.shape[1] == 5

    def test_class_separation(self):
        """class_separation should affect distance between means."""
        loader_close = GaussianGenerator(class_separation=1.0, shuffle=False)
        loader_far = GaussianGenerator(class_separation=10.0, shuffle=False)

        # Means should be further apart with higher separation
        mean_dist_close = np.linalg.norm(
            loader_close.means[1] - loader_close.means[0]
        )
        mean_dist_far = np.linalg.norm(
            loader_far.means[1] - loader_far.means[0]
        )

        assert mean_dist_far > mean_dist_close


class TestGaussianMeans:
    """Tests for mean configuration."""

    def test_custom_means(self):
        """Custom means should be used directly."""
        custom_means = [[1, 2], [5, 6]]
        loader = GaussianGenerator(means=custom_means)

        np.testing.assert_array_equal(loader.means[0], [1, 2])
        np.testing.assert_array_equal(loader.means[1], [5, 6])

    def test_means_infer_n_features(self):
        """n_features should be inferred from custom means."""
        loader = GaussianGenerator(means=[[0, 0, 0], [1, 1, 1]])

        assert loader.n_features == 3
        assert loader.get_X().shape[1] == 3

    def test_means_wrong_count_raises(self):
        """means with != 2 classes should raise ValueError."""
        with pytest.raises(ValueError, match="must have exactly 2 classes"):
            GaussianGenerator(means=[[0, 0], [1, 1], [2, 2]])

    def test_auto_means_at_origin_and_diagonal(self):
        """Auto-generated means should be at origin and diagonal."""
        loader = GaussianGenerator(n_features=3, class_separation=5.0)

        np.testing.assert_array_equal(loader.means[0], [0, 0, 0])
        np.testing.assert_array_equal(loader.means[1], [5, 5, 5])


class TestGaussianCovarianceTypes:
    """Tests for different covariance types."""

    def test_cov_type_spherical(self):
        """Spherical covariance should be identity * scale."""
        loader = GaussianGenerator(cov_type='spherical', cov_scale=2.0)

        expected = np.eye(2) * 2.0
        np.testing.assert_array_equal(loader.covs[0], expected)
        np.testing.assert_array_equal(loader.covs[1], expected)

    def test_cov_type_diagonal(self):
        """Diagonal covariance should have zeros off-diagonal."""
        loader = GaussianGenerator(cov_type='diagonal', n_features=3)

        for cov in loader.covs:
            # Check diagonal is non-zero
            assert np.all(np.diag(cov) > 0)
            # Check off-diagonal is zero
            off_diag = cov - np.diag(np.diag(cov))
            np.testing.assert_array_equal(off_diag, np.zeros((3, 3)))

    def test_cov_type_symmetric(self):
        """Symmetric covariance should have uniform off-diagonal."""
        loader = GaussianGenerator(
            cov_type='symmetric',
            cov_scale=1.0,
            cov_correlation=0.5
        )

        cov = loader.covs[0]
        # Diagonal should be scale
        np.testing.assert_array_almost_equal(np.diag(cov), [1.0, 1.0])
        # Off-diagonal should be correlation * scale
        assert cov[0, 1] == 0.5
        assert cov[1, 0] == 0.5

    def test_cov_type_random_positive_definite(self):
        """Random covariance should be positive definite."""
        loader = GaussianGenerator(cov_type='random', n_features=4)

        for cov in loader.covs:
            eigenvalues = np.linalg.eigvals(cov)
            assert np.all(eigenvalues > 0), "Covariance must be positive definite"

    def test_cov_type_invalid_raises(self):
        """Invalid cov_type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown cov_type"):
            GaussianGenerator(cov_type='invalid')


class TestGaussianCovarianceScale:
    """Tests for covariance scaling."""

    def test_cov_scale_single_value(self):
        """Single cov_scale should apply to both classes."""
        loader = GaussianGenerator(cov_type='spherical', cov_scale=3.0)

        assert np.trace(loader.covs[0]) == 6.0  # 3.0 * 2 features
        assert np.trace(loader.covs[1]) == 6.0

    def test_cov_scale_per_class(self):
        """Per-class cov_scale should apply differently."""
        loader = GaussianGenerator(
            cov_type='spherical',
            cov_scale=[1.0, 4.0]
        )

        assert np.trace(loader.covs[0]) == 2.0  # 1.0 * 2 features
        assert np.trace(loader.covs[1]) == 8.0  # 4.0 * 2 features

    def test_cov1_scaler_default_no_effect(self):
        """cov1_scaler=1.0 should leave both covs equal (spherical default)."""
        loader = GaussianGenerator(cov_type='spherical', cov1_scaler=1.0)

        np.testing.assert_array_equal(loader.covs[0], loader.covs[1])

    def test_cov1_scaler_doubles_cov(self):
        """cov1_scaler=2.0 should make cov1 == 2 * cov0 element-wise."""
        loader = GaussianGenerator(cov_type='spherical', cov1_scaler=2.0)

        np.testing.assert_array_equal(loader.covs[1], 2.0 * loader.covs[0])

    def test_cov1_scaler_with_existing_scale(self):
        """cov1_scaler should be applied relative to cov0, not independent cov1."""
        loader = GaussianGenerator(
            cov_type='spherical',
            cov_scale=0.5,
            cov1_scaler=3.0
        )

        # cov0 = 0.5 * I; cov1 should be 3 * cov0 = 1.5 * I
        np.testing.assert_array_equal(loader.covs[1], 3.0 * loader.covs[0])


class TestGaussianCustomCovariance:
    """Tests for custom covariance matrices."""

    def test_custom_covs(self):
        """Custom covariance matrices should be used directly."""
        custom_covs = [
            [[1, 0.5], [0.5, 1]],
            [[2, -0.3], [-0.3, 2]]
        ]
        loader = GaussianGenerator(covs=custom_covs)

        np.testing.assert_array_equal(loader.covs[0], custom_covs[0])
        np.testing.assert_array_equal(loader.covs[1], custom_covs[1])

    def test_custom_covs_wrong_count_raises(self):
        """covs with != 2 matrices should raise ValueError."""
        with pytest.raises(ValueError, match="must have exactly 2 matrices"):
            GaussianGenerator(covs=[[[1, 0], [0, 1]]])

    def test_custom_covs_sets_cov_type_custom(self):
        """Custom covs should set cov_type to 'custom'."""
        loader = GaussianGenerator(covs=[[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
        assert loader.cov_type == 'custom'


class TestGaussianNumSamplesValidation:
    """Tests for num_samples validation."""

    def test_num_samples_wrong_count_raises(self):
        """num_samples with != 2 values should raise ValueError."""
        with pytest.raises(ValueError, match="must have 2 values"):
            GaussianGenerator(num_samples=[100, 200, 300])


class TestGaussianMetadata:
    """Tests for dataset metadata."""

    def test_has_description(self):
        """Dataset should have a description."""
        loader = GaussianGenerator()
        desc = loader.get_description()

        assert 'Gaussian' in desc
        assert 'samples' in desc.lower()

    def test_has_feature_names(self):
        """Dataset should have feature names."""
        loader = GaussianGenerator(n_features=3)
        names = loader.get_feature_names()

        assert len(names) == 3
        assert 'Feature 1' in names

    def test_has_label_names(self):
        """Dataset should have label names."""
        loader = GaussianGenerator()
        labels = loader.get_label_names()

        assert labels == ['Class 0', 'Class 1']

    def test_name_parameter(self):
        """Custom name should be used."""
        loader = GaussianGenerator(name='My Custom Gaussian')
        assert loader.name == 'My Custom Gaussian'


class TestGaussianTrainTestSplit:
    """Tests for train/test splitting."""

    def test_train_test_split_works(self):
        """Train/test split should work correctly."""
        loader = GaussianGenerator(num_samples=200)
        train, test = loader.get_train_test_split()

        assert 'X' in train and 'y' in train
        assert 'X' in test and 'y' in test
        assert len(train['y']) + len(test['y']) == 200

    def test_train_test_preserves_classes(self):
        """Both splits should contain both classes."""
        loader = GaussianGenerator(num_samples=200)
        train, test = loader.get_train_test_split()

        assert set(np.unique(train['y'])) == {0, 1}
        assert set(np.unique(test['y'])) == {0, 1}


class TestGaussianReproducibility:
    """Tests for reproducibility."""

    def test_same_seed_same_data(self):
        """Same seed should produce same data."""
        loader1 = GaussianGenerator(set_seed=42, shuffle=False)
        loader2 = GaussianGenerator(set_seed=42, shuffle=False)

        np.testing.assert_array_equal(loader1.get_X(), loader2.get_X())
        np.testing.assert_array_equal(loader1.get_y(), loader2.get_y())


class TestGaussianClassesIndependent:
    """The two classes must be independent draws, not copies of one another.

    Regression test: load_data used to re-seed before drawing each class, so every
    class got the same standard-normal draws and class 1 was an exact translate of
    class 0 (a scaled translate when cov1_scaler != 1).
    """

    @pytest.mark.parametrize('cov1_scaler', [1.0, 3.0])
    def test_class_one_is_not_a_transform_of_class_zero(self, cov1_scaler):
        loader = GaussianGenerator(n_features=3, class_separation=2.0, num_samples=[50, 50],
                                   cov1_scaler=cov1_scaler, shuffle=False, set_seed=42)
        data = loader.load_data()
        X0 = data['X'][data['y'] == 0] - loader.means[0]
        X1 = data['X'][data['y'] == 1] - loader.means[1]
        # a translate would leave X1 == X0; a scaled one X1 == sqrt(r) * X0
        assert not np.allclose(X1, X0)
        assert not np.allclose(X1, np.sqrt(cov1_scaler)*X0)
        # and they should not be correlated row for row either
        assert abs(np.corrcoef(X0.ravel(), X1.ravel())[0, 1]) < 0.5

    def test_normal_loader_classes_are_independent(self):
        from data_loaders.loaders.synthetic_generators.normal import NormalDataLoader
        loader = NormalDataLoader(num_train=100, num_test=100, train_ratio=1,
                                  shuffle=False, set_seed=42)
        data = loader.load_data()
        n = min(np.sum(data['y'] == 0), np.sum(data['y'] == 1))
        X0 = data['X'][data['y'] == 0][:n] - np.array(loader.m1)
        X1 = data['X'][data['y'] == 1][:n] - np.array(loader.m2)
        assert not np.allclose(X1, X0)
