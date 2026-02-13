"""Pytest fixtures and configuration."""

import numpy as np
import pytest
import torch

from adaptive_feature_interaction_learning_with_dynamic_gating.models.model import (
    AdaptiveFeatureInteractionModel,
)
from adaptive_feature_interaction_learning_with_dynamic_gating.data.preprocessing import (
    TabularPreprocessor,
)


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


@pytest.fixture
def sample_regression_data(random_seed):
    """Generate sample regression data."""
    np.random.seed(random_seed)
    X = np.random.randn(100, 20)
    y = np.random.randn(100)
    return X, y


@pytest.fixture
def sample_classification_data(random_seed):
    """Generate sample classification data."""
    np.random.seed(random_seed)
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 2, size=100)
    return X, y


@pytest.fixture
def sample_multiclass_data(random_seed):
    """Generate sample multiclass classification data."""
    np.random.seed(random_seed)
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 5, size=100)
    return X, y


@pytest.fixture
def regression_model(device):
    """Create a regression model for testing."""
    model = AdaptiveFeatureInteractionModel(
        input_dim=20,
        hidden_dims=[64, 32],
        output_dim=1,
        task_type="regression",
        use_gating=True,
        dropout_rate=0.2,
    )
    return model.to(device)


@pytest.fixture
def classification_model(device):
    """Create a binary classification model for testing."""
    model = AdaptiveFeatureInteractionModel(
        input_dim=20,
        hidden_dims=[64, 32],
        output_dim=1,
        task_type="classification",
        use_gating=True,
        dropout_rate=0.2,
    )
    return model.to(device)


@pytest.fixture
def multiclass_model(device):
    """Create a multiclass classification model for testing."""
    model = AdaptiveFeatureInteractionModel(
        input_dim=20,
        hidden_dims=[64, 32],
        output_dim=5,
        task_type="classification",
        use_gating=True,
        dropout_rate=0.2,
    )
    return model.to(device)


@pytest.fixture
def preprocessor():
    """Create a preprocessor for testing."""
    return TabularPreprocessor(
        scaling_method="standard",
        handle_outliers=True,
    )
