"""Tests for data loading and preprocessing."""

import numpy as np
import pytest

from adaptive_feature_interaction_learning_with_dynamic_gating.data.loader import (
    generate_synthetic_dataset,
    load_dataset,
    get_data_loaders,
)
from adaptive_feature_interaction_learning_with_dynamic_gating.data.preprocessing import (
    TabularPreprocessor,
)


class TestDataLoading:
    """Test data loading utilities."""

    def test_generate_synthetic_regression(self, random_seed):
        """Test synthetic regression dataset generation."""
        X, y = generate_synthetic_dataset(
            n_samples=100,
            n_features=20,
            n_informative=10,
            task_type="regression",
            random_state=random_seed,
        )

        assert X.shape == (100, 20)
        assert y.shape == (100,)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_generate_synthetic_classification(self, random_seed):
        """Test synthetic classification dataset generation."""
        X, y = generate_synthetic_dataset(
            n_samples=100,
            n_features=20,
            n_informative=10,
            task_type="classification",
            n_classes=3,
            random_state=random_seed,
        )

        assert X.shape == (100, 20)
        assert y.shape == (100,)
        assert len(np.unique(y)) <= 3
        assert y.min() >= 0

    def test_load_dataset_synthetic(self, random_seed):
        """Test loading synthetic dataset."""
        X, y = load_dataset(
            dataset_name="synthetic",
            task_type="regression",
            n_samples=50,
            n_features=10,
            random_state=random_seed,
        )

        assert X.shape[0] == 50
        assert X.shape[1] == 10

    def test_get_data_loaders(self, sample_regression_data, random_seed):
        """Test data loader creation."""
        X, y = sample_regression_data

        data_loaders = get_data_loaders(
            X, y,
            batch_size=16,
            val_split=0.2,
            test_split=0.2,
            random_state=random_seed,
        )

        assert "train" in data_loaders
        assert "val" in data_loaders
        assert "test" in data_loaders

        # Check batch sizes
        for batch in data_loaders["train"]:
            data, target = batch
            assert data.shape[0] <= 16
            break


class TestPreprocessing:
    """Test preprocessing utilities."""

    def test_preprocessor_fit_transform(self, sample_regression_data, preprocessor):
        """Test preprocessor fit and transform."""
        X, _ = sample_regression_data

        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape == X.shape
        assert preprocessor.is_fitted
        # Check standardization worked
        assert np.abs(X_transformed.mean()) < 1.0
        assert np.abs(X_transformed.std() - 1.0) < 0.5

    def test_preprocessor_transform_unfitted(self, sample_regression_data):
        """Test that transform raises error when not fitted."""
        preprocessor = TabularPreprocessor()
        X, _ = sample_regression_data

        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(X)

    def test_preprocessor_handles_nan(self, preprocessor):
        """Test that preprocessor handles NaN values."""
        X = np.random.randn(100, 10)
        X[0, 0] = np.nan
        X[5, 3] = np.nan

        X_transformed = preprocessor.fit_transform(X)

        assert not np.isnan(X_transformed).any()

    def test_preprocessor_feature_stats(self, sample_regression_data, preprocessor):
        """Test feature statistics computation."""
        X, _ = sample_regression_data

        stats = preprocessor.get_feature_stats(X)

        assert "n_features" in stats
        assert "n_samples" in stats
        assert stats["n_features"] == X.shape[1]
        assert stats["n_samples"] == X.shape[0]

    def test_preprocessor_inverse_transform(self, sample_regression_data):
        """Test inverse transform."""
        X, _ = sample_regression_data

        # Use preprocessor without outlier handling for exact inverse
        preprocessor_no_outliers = TabularPreprocessor(
            scaling_method="standard",
            handle_outliers=False,
        )
        X_transformed = preprocessor_no_outliers.fit_transform(X)
        X_recovered = preprocessor_no_outliers.inverse_transform(X_transformed)

        # Should be close to original (within numerical precision)
        assert np.allclose(X, X_recovered, atol=1e-5)
