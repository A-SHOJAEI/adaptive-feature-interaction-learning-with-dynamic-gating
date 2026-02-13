"""Tests for training utilities."""

import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from adaptive_feature_interaction_learning_with_dynamic_gating.training.trainer import (
    AdaptiveTrainer,
)
from adaptive_feature_interaction_learning_with_dynamic_gating.evaluation.metrics import (
    compute_metrics,
    compute_confidence_intervals,
    compute_relative_improvement,
)


class TestAdaptiveTrainer:
    """Test adaptive trainer."""

    @pytest.fixture
    def train_loader(self, sample_regression_data, device):
        """Create a training data loader."""
        X, y = sample_regression_data
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    @pytest.fixture
    def val_loader(self, sample_regression_data, device):
        """Create a validation data loader."""
        X, y = sample_regression_data
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=16, shuffle=False)

    def test_trainer_initialization(self, regression_model, device):
        """Test trainer initialization."""
        optimizer = optim.Adam(regression_model.parameters(), lr=0.001)

        trainer = AdaptiveTrainer(
            model=regression_model,
            optimizer=optimizer,
            device=device,
            task_type="regression",
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.device == device

    def test_trainer_train_epoch(self, regression_model, train_loader, device):
        """Test training for one epoch."""
        optimizer = optim.Adam(regression_model.parameters(), lr=0.001)

        trainer = AdaptiveTrainer(
            model=regression_model,
            optimizer=optimizer,
            device=device,
            task_type="regression",
        )

        metrics = trainer.train_epoch(train_loader, epoch=0, total_epochs=10)

        assert "train_loss" in metrics
        assert metrics["train_loss"] > 0

    def test_trainer_validate(self, regression_model, val_loader, device):
        """Test validation."""
        optimizer = optim.Adam(regression_model.parameters(), lr=0.001)

        trainer = AdaptiveTrainer(
            model=regression_model,
            optimizer=optimizer,
            device=device,
            task_type="regression",
        )

        metrics = trainer.validate(val_loader)

        assert "val_loss" in metrics
        assert metrics["val_loss"] > 0

    def test_trainer_full_training(
        self,
        regression_model,
        train_loader,
        val_loader,
        device,
        tmp_path,
    ):
        """Test full training loop."""
        optimizer = optim.Adam(regression_model.parameters(), lr=0.01)

        trainer = AdaptiveTrainer(
            model=regression_model,
            optimizer=optimizer,
            device=device,
            task_type="regression",
            early_stopping_patience=5,
        )

        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5,
            checkpoint_dir=tmp_path,
        )

        assert "history" in results
        assert "best_epoch" in results
        assert "best_val_loss" in results
        assert len(results["history"]["train_loss"]) <= 5

        # Check checkpoint was saved
        assert (tmp_path / "best_model.pt").exists()

    def test_trainer_checkpoint_save_load(
        self,
        regression_model,
        device,
        tmp_path,
    ):
        """Test checkpoint saving and loading."""
        optimizer = optim.Adam(regression_model.parameters(), lr=0.001)

        trainer = AdaptiveTrainer(
            model=regression_model,
            optimizer=optimizer,
            device=device,
            task_type="regression",
        )

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)


class TestEvaluationMetrics:
    """Test evaluation metrics."""

    def test_regression_metrics(self, sample_regression_data):
        """Test regression metrics computation."""
        X, y_true = sample_regression_data
        y_pred = y_true + 0.1  # Add small noise

        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            task_type="regression",
        )

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["mse"] > 0

    def test_classification_metrics(self, sample_classification_data):
        """Test classification metrics computation."""
        X, y_true = sample_classification_data
        y_pred = y_true.copy()
        y_pred[:10] = 1 - y_pred[:10]  # Flip some predictions

        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            task_type="classification",
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_confidence_intervals(self, sample_regression_data):
        """Test confidence interval computation."""
        X, y_true = sample_regression_data
        y_pred = y_true + 0.1

        ci_results = compute_confidence_intervals(
            y_true=y_true,
            y_pred=y_pred,
            task_type="regression",
            n_bootstraps=100,  # Fewer for testing
            confidence_level=0.95,
        )

        assert len(ci_results) > 0
        for metric_name, (lower, mean, upper) in ci_results.items():
            # Allow small numerical tolerance for floating point errors
            assert lower <= mean + 1e-10, f"{metric_name}: lower={lower}, mean={mean}"
            assert mean <= upper + 1e-10, f"{metric_name}: mean={mean}, upper={upper}"

    def test_relative_improvement(self):
        """Test relative improvement computation."""
        baseline_metrics = {"rmse": 1.0, "mae": 0.8}
        model_metrics = {"rmse": 0.9, "mae": 0.7}

        improvement = compute_relative_improvement(
            baseline_metrics=baseline_metrics,
            model_metrics=model_metrics,
            primary_metric="rmse",
        )

        assert improvement > 0  # Model is better (lower RMSE)
        assert 0.05 < improvement < 0.15  # ~10% improvement
