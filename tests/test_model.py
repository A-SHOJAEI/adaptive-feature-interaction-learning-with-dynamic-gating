"""Tests for model components and architecture."""

import numpy as np
import pytest
import torch

from adaptive_feature_interaction_learning_with_dynamic_gating.models.model import (
    AdaptiveFeatureInteractionModel,
)
from adaptive_feature_interaction_learning_with_dynamic_gating.models.components import (
    DynamicGatingLayer,
    HierarchicalFeatureInteraction,
    ConfidenceWeightedLoss,
    GradientFlowAnalyzer,
)


class TestDynamicGatingLayer:
    """Test dynamic gating layer."""

    def test_gating_layer_forward(self, device):
        """Test forward pass through gating layer."""
        layer = DynamicGatingLayer(
            input_dim=20,
            num_gates=64,
        ).to(device)

        x = torch.randn(16, 20).to(device)
        gates, gate_info = layer(x, training_progress=0.5)

        assert gates.shape == (16, 64)
        assert "gate_probs" in gate_info
        assert "confidence" in gate_info
        assert "sparsity_ratio" in gate_info

    def test_gating_temperature_annealing(self, device):
        """Test temperature annealing."""
        layer = DynamicGatingLayer(
            input_dim=20,
            num_gates=64,
            gate_temperature=1.0,
            temperature_decay=0.9,
        ).to(device)

        initial_temp = layer.current_temperature

        layer.anneal_temperature()
        layer.anneal_temperature()

        assert layer.current_temperature < initial_temp


class TestHierarchicalFeatureInteraction:
    """Test hierarchical feature interaction module."""

    def test_interaction_forward(self, device):
        """Test forward pass through interaction module."""
        module = HierarchicalFeatureInteraction(
            input_dim=20,
            hidden_dim=64,
            max_order=3,
            use_gating=True,
        ).to(device)

        x = torch.randn(16, 20).to(device)
        output, gate_info = module(x, training_progress=0.5)

        assert output.shape == (16, 64)
        assert gate_info is not None

    def test_interaction_without_gating(self, device):
        """Test interaction module without gating."""
        module = HierarchicalFeatureInteraction(
            input_dim=20,
            hidden_dim=64,
            max_order=3,
            use_gating=False,
        ).to(device)

        x = torch.randn(16, 20).to(device)
        output, gate_info = module(x, training_progress=0.5)

        assert output.shape == (16, 64)
        assert gate_info is None


class TestConfidenceWeightedLoss:
    """Test confidence-weighted loss function."""

    def test_mse_loss(self, device):
        """Test MSE-based confidence loss."""
        criterion = ConfidenceWeightedLoss(
            base_loss="mse",
            confidence_scale=1.0,
        )

        predictions = torch.randn(32, 1).to(device)
        targets = torch.randn(32, 1).to(device)

        loss, loss_info = criterion(predictions, targets)

        assert loss.item() > 0
        assert "base_loss" in loss_info
        assert "weighted_loss" in loss_info

    def test_cross_entropy_loss(self, device):
        """Test cross-entropy based confidence loss."""
        criterion = ConfidenceWeightedLoss(
            base_loss="cross_entropy",
            confidence_scale=1.0,
        )

        predictions = torch.randn(32, 5).to(device)
        targets = torch.randint(0, 5, (32,)).to(device)

        loss, loss_info = criterion(predictions, targets)

        assert loss.item() > 0
        assert "base_loss" in loss_info
        assert "mean_confidence" in loss_info


class TestGradientFlowAnalyzer:
    """Test gradient flow analyzer."""

    def test_analyze_gradients(self, regression_model, device):
        """Test gradient analysis."""
        analyzer = GradientFlowAnalyzer()

        # Forward pass
        x = torch.randn(16, 20).to(device)
        y = torch.randn(16, 1).to(device)

        predictions, _, _ = regression_model(x)
        loss = torch.nn.functional.mse_loss(predictions, y)

        # Backward pass
        loss.backward()

        # Analyze gradients
        stats = analyzer.analyze_gradients(regression_model)

        assert "total_gradient_norm" in stats
        assert "mean_gradient_norm" in stats
        assert stats["total_gradient_norm"] > 0


class TestAdaptiveFeatureInteractionModel:
    """Test the main model."""

    def test_regression_model_forward(self, regression_model, device):
        """Test forward pass for regression."""
        x = torch.randn(16, 20).to(device)

        predictions, confidence, gate_info = regression_model(
            x,
            return_confidence=True,
            return_gate_info=True,
        )

        assert predictions.shape == (16, 1)
        assert confidence.shape == (16, 1)
        assert gate_info is not None

    def test_classification_model_forward(self, classification_model, device):
        """Test forward pass for binary classification."""
        x = torch.randn(16, 20).to(device)

        predictions, confidence, gate_info = classification_model(
            x,
            return_confidence=True,
            return_gate_info=True,
        )

        assert predictions.shape == (16, 1)
        assert confidence is not None

    def test_multiclass_model_forward(self, multiclass_model, device):
        """Test forward pass for multiclass classification."""
        x = torch.randn(16, 20).to(device)

        predictions, confidence, gate_info = multiclass_model(
            x,
            return_confidence=True,
            return_gate_info=True,
        )

        assert predictions.shape == (16, 5)
        assert confidence is not None

    def test_model_predict(self, regression_model, device):
        """Test prediction method."""
        x = torch.randn(16, 20).to(device)

        predictions = regression_model.predict(x)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (16, 1)

    def test_model_predict_proba(self, multiclass_model, device):
        """Test probability prediction."""
        x = torch.randn(16, 20).to(device)

        probabilities = multiclass_model.predict_proba(x)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (16, 5)
        # Check probabilities sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)

    def test_model_sparsity_stats(self, regression_model):
        """Test sparsity statistics."""
        stats = regression_model.get_sparsity_stats()

        assert "sparsity_ratio" in stats
        assert "gate_entropy" in stats
        assert 0 <= stats["sparsity_ratio"] <= 1

    def test_model_training_progress_update(self, regression_model):
        """Test training progress update."""
        regression_model.update_training_progress(epoch=50, total_epochs=100)

        assert regression_model.current_epoch == 50
        assert regression_model.total_epochs == 100

    def test_model_config(self, regression_model):
        """Test model configuration retrieval."""
        config = regression_model.get_config()

        assert "input_dim" in config
        assert "hidden_dims" in config
        assert "task_type" in config
        assert config["input_dim"] == 20
        assert config["task_type"] == "regression"

    def test_model_parameter_count(self, regression_model):
        """Test parameter counting."""
        num_params = regression_model.count_parameters()

        assert num_params > 0
        assert isinstance(num_params, int)

    def test_model_without_gating(self, device):
        """Test model without gating mechanism."""
        model = AdaptiveFeatureInteractionModel(
            input_dim=20,
            hidden_dims=[64, 32],
            output_dim=1,
            task_type="regression",
            use_gating=False,
        ).to(device)

        x = torch.randn(16, 20).to(device)
        predictions, confidence, gate_info = model(
            x,
            return_confidence=True,
            return_gate_info=True,
        )

        assert predictions.shape == (16, 1)

    def test_model_backward_pass(self, regression_model, device):
        """Test backward pass works correctly."""
        regression_model.train()  # Set to training mode
        x = torch.randn(16, 20).to(device)
        y = torch.randn(16, 1).to(device)

        predictions, _, _ = regression_model(x)
        loss = torch.nn.functional.mse_loss(predictions, y)

        loss.backward()

        # Check that at least some gradients exist (not all params may have gradients in eval mode)
        grad_count = sum(1 for param in regression_model.parameters() if param.grad is not None)
        assert grad_count > 0
