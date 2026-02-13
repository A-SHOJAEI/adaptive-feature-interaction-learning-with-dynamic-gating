"""Training loop with advanced features for adaptive model training."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from adaptive_feature_interaction_learning_with_dynamic_gating.models.components import (
    ConfidenceWeightedLoss,
    GradientFlowAnalyzer,
)

logger = logging.getLogger(__name__)


class AdaptiveTrainer:
    """Trainer with adaptive learning and early stopping.

    Features:
    - Confidence-weighted loss
    - Learning rate scheduling
    - Early stopping with patience
    - Gradient clipping
    - Mixed precision training
    - Comprehensive logging

    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        device: Device to train on
        task_type: 'regression' or 'classification'
        use_confidence_loss: Whether to use confidence-weighted loss
        gradient_clip_value: Max gradient norm for clipping
        early_stopping_patience: Epochs to wait before early stopping
        lr_scheduler_type: 'cosine', 'reduce_on_plateau', or None
        use_mixed_precision: Whether to use automatic mixed precision
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        task_type: str = "regression",
        use_confidence_loss: bool = True,
        gradient_clip_value: float = 5.0,
        early_stopping_patience: int = 15,
        lr_scheduler_type: Optional[str] = "cosine",
        use_mixed_precision: bool = False,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.task_type = task_type
        self.use_confidence_loss = use_confidence_loss
        self.gradient_clip_value = gradient_clip_value
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_type = lr_scheduler_type
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()

        # Loss function
        if use_confidence_loss:
            base_loss = "cross_entropy" if task_type == "classification" else "mse"
            self.criterion = ConfidenceWeightedLoss(
                base_loss=base_loss,
                confidence_scale=1.0,
                gradient_clip=gradient_clip_value,
            )
        else:
            if task_type == "classification":
                self.criterion = nn.CrossEntropyLoss()
            else:
                self.criterion = nn.MSELoss()

        # Learning rate scheduler
        self.scheduler = None

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None

        # Gradient flow analyzer
        self.gradient_analyzer = GradientFlowAnalyzer()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "sparsity_ratio": [],
            "gate_entropy": [],
        }

    def initialize_scheduler(self, total_epochs: int, steps_per_epoch: int) -> None:
        """Initialize learning rate scheduler.

        Args:
            total_epochs: Total number of training epochs
            steps_per_epoch: Number of steps per epoch
        """
        if self.lr_scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs,
                eta_min=1e-6,
            )
            logger.info("Initialized cosine annealing scheduler")
        elif self.lr_scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True,
            )
            logger.info("Initialized reduce on plateau scheduler")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            total_epochs: Total number of epochs

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.model.update_training_progress(epoch, total_epochs)

        total_loss = 0.0
        total_base_loss = 0.0
        total_confidence = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Handle target shape
            if self.task_type == "regression" and target.dim() == 1:
                target = target.unsqueeze(-1)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions, confidence, gate_info = self.model(
                        data,
                        return_confidence=True,
                        return_gate_info=True,
                    )

                    # Compute loss
                    if self.use_confidence_loss:
                        loss, loss_info = self.criterion(predictions, target, confidence)
                    else:
                        if self.task_type == "classification":
                            loss = self.criterion(predictions, target.squeeze())
                        else:
                            loss = self.criterion(predictions, target)
                        loss_info = {"base_loss": loss.item(), "weighted_loss": loss.item()}

                # Backward pass with scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_value
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                predictions, confidence, gate_info = self.model(
                    data,
                    return_confidence=True,
                    return_gate_info=True,
                )

                # Compute loss
                if self.use_confidence_loss:
                    loss, loss_info = self.criterion(predictions, target, confidence)
                else:
                    if self.task_type == "classification":
                        loss = self.criterion(predictions, target.squeeze())
                    else:
                        loss = self.criterion(predictions, target)
                    loss_info = {"base_loss": loss.item(), "weighted_loss": loss.item()}

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_value
                )

                # Optimizer step
                self.optimizer.step()

            # Accumulate metrics
            total_loss += loss_info["weighted_loss"]
            total_base_loss += loss_info["base_loss"]
            if confidence is not None:
                total_confidence += confidence.mean().item()
            num_batches += 1

        # Compute epoch metrics
        metrics = {
            "train_loss": total_loss / num_batches,
            "base_loss": total_base_loss / num_batches,
            "mean_confidence": total_confidence / num_batches,
        }

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_confidence = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Handle target shape
                if self.task_type == "regression" and target.dim() == 1:
                    target = target.unsqueeze(-1)

                # Forward pass
                predictions, confidence, _ = self.model(
                    data,
                    return_confidence=True,
                )

                # Compute loss
                if self.task_type == "classification":
                    loss = nn.functional.cross_entropy(predictions, target.squeeze())
                else:
                    loss = nn.functional.mse_loss(predictions, target)

                total_loss += loss.item()
                if confidence is not None:
                    total_confidence += confidence.mean().item()
                num_batches += 1

                # Store predictions for metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(target.cpu())

        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        metrics = {
            "val_loss": total_loss / num_batches,
            "mean_confidence": total_confidence / num_batches,
        }

        # Task-specific metrics
        if self.task_type == "classification":
            pred_classes = all_predictions.argmax(dim=-1)
            accuracy = (pred_classes == all_targets.squeeze()).float().mean().item()
            metrics["accuracy"] = accuracy
        else:
            mse = ((all_predictions - all_targets) ** 2).mean().item()
            rmse = np.sqrt(mse)
            metrics["rmse"] = rmse

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary of training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        # Initialize scheduler
        self.initialize_scheduler(num_epochs, len(train_loader))

        # Create checkpoint directory
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train one epoch
            train_metrics = self.train_epoch(train_loader, epoch, num_epochs)

            # Validate
            val_metrics = self.validate(val_loader)

            # Get sparsity stats
            sparsity_stats = self.model.get_sparsity_stats()

            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if self.lr_scheduler_type == "reduce_on_plateau":
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Update history
            self.training_history["train_loss"].append(train_metrics["train_loss"])
            self.training_history["val_loss"].append(val_metrics["val_loss"])
            self.training_history["learning_rate"].append(current_lr)
            self.training_history["sparsity_ratio"].append(sparsity_stats["sparsity_ratio"])
            self.training_history["gate_entropy"].append(sparsity_stats["gate_entropy"])

            # Early stopping check
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save best model
                if checkpoint_dir is not None:
                    self.save_checkpoint(checkpoint_dir / "best_model.pt")
            else:
                self.patience_counter += 1

            # Log progress
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} ({epoch_time:.2f}s) - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"LR: {current_lr:.6f}, "
                f"Sparsity: {sparsity_stats['sparsity_ratio']:.3f}"
            )

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best epoch: {self.best_epoch + 1}"
                )
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")

        # Load best model
        if checkpoint_dir is not None:
            self.load_checkpoint(checkpoint_dir / "best_model.pt")

        return {
            "history": self.training_history,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_time": total_time,
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint.get("training_history", self.training_history)

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Checkpoint loaded from {path}")
