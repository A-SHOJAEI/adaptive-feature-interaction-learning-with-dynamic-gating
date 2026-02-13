"""Core model implementation for adaptive feature interaction learning."""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_feature_interaction_learning_with_dynamic_gating.models.components import (
    HierarchicalFeatureInteraction,
    DynamicGatingLayer,
)

logger = logging.getLogger(__name__)


class AdaptiveFeatureInteractionModel(nn.Module):
    """Adaptive Feature Interaction Model with Dynamic Gating.

    Novel contributions:
    1. Hierarchical feature interactions that evolve during training
    2. Confidence-weighted feature selection
    3. Dynamic gating that adjusts interaction complexity based on convergence

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (1 for regression, num_classes for classification)
        task_type: 'regression' or 'classification'
        use_gating: Whether to use dynamic gating
        dropout_rate: Dropout probability
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        output_dim: int = 1,
        task_type: str = "regression",
        use_gating: bool = True,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.task_type = task_type
        self.use_gating = use_gating
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Input embedding layer
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else nn.Identity(),
            nn.Dropout(dropout_rate),
        )

        # Hierarchical feature interaction module
        self.feature_interaction = HierarchicalFeatureInteraction(
            input_dim=input_dim,
            hidden_dim=hidden_dims[0],
            max_order=3,
            use_gating=use_gating,
        )

        # Deep layers for further processing
        deep_layers = []
        for i in range(len(hidden_dims) - 1):
            deep_layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[i + 1]) if use_batch_norm else nn.Identity(),
                nn.Dropout(dropout_rate),
            ])
        self.deep_layers = nn.Sequential(*deep_layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Training state
        self.current_epoch = 0
        self.total_epochs = 100  # Will be updated during training

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_confidence: bool = False,
        return_gate_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_confidence: Whether to return confidence scores
            return_gate_info: Whether to return gating information

        Returns:
            Tuple of (predictions, confidence, gate_info)
            - predictions: Model predictions
            - confidence: Confidence scores (if return_confidence=True)
            - gate_info: Gating statistics (if return_gate_info=True)
        """
        batch_size = x.size(0)

        # Compute training progress
        training_progress = self.current_epoch / max(self.total_epochs, 1)

        # Input embedding
        h_embed = self.input_embedding(x)

        # Hierarchical feature interactions with dynamic gating
        h_interact, gate_info = self.feature_interaction(x, training_progress)

        # Combine embedded features with interaction features
        h_combined = h_embed + h_interact

        # Deep processing
        h_deep = self.deep_layers(h_combined)

        # Predictions
        predictions = self.output_layer(h_deep)

        # Confidence estimation
        confidence = None
        if return_confidence or self.training:
            confidence = self.confidence_head(h_deep)

        # Format output based on task type
        if self.task_type == "classification" and self.output_dim > 1:
            predictions = F.log_softmax(predictions, dim=-1)

        # Prepare return values
        gate_info_out = gate_info if return_gate_info else None

        return predictions, confidence, gate_info_out

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Make predictions on input data.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Predictions as numpy array
        """
        self.eval()
        with torch.no_grad():
            predictions, _, _ = self.forward(x, return_confidence=False)

            if self.task_type == "classification" and self.output_dim > 1:
                # Convert log probabilities to class predictions
                predictions = predictions.argmax(dim=-1)
            elif self.task_type == "classification":
                # Binary classification
                predictions = (predictions > 0).long()

        return predictions.cpu().numpy()

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Predict class probabilities (for classification only).

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Class probabilities as numpy array

        Raises:
            ValueError: If task_type is not classification
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification tasks")

        self.eval()
        with torch.no_grad():
            predictions, _, _ = self.forward(x, return_confidence=False)
            if self.output_dim > 1:
                probs = torch.exp(predictions)  # Convert log_softmax to probabilities
            else:
                probs = torch.sigmoid(predictions)

        return probs.cpu().numpy()

    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get sparsity statistics from gating mechanism.

        Returns:
            Dictionary of sparsity metrics
        """
        if not self.use_gating:
            return {"sparsity_ratio": 0.0, "gate_entropy": 0.0}

        # Create dummy input to get gate statistics
        dummy_input = torch.zeros(1, self.input_dim)
        if next(self.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()

        self.eval()
        with torch.no_grad():
            _, _, gate_info = self.forward(
                dummy_input,
                return_gate_info=True
            )

        if gate_info is None:
            return {"sparsity_ratio": 0.0, "gate_entropy": 0.0}

        return {
            "sparsity_ratio": gate_info["sparsity_ratio"].item(),
            "gate_entropy": gate_info["gate_entropy"].item(),
        }

    def update_training_progress(self, epoch: int, total_epochs: int) -> None:
        """Update training progress for adaptive gating.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of training epochs
        """
        self.current_epoch = epoch
        self.total_epochs = total_epochs

        # Anneal temperature in gating layers
        if self.use_gating:
            for module in self.modules():
                if isinstance(module, DynamicGatingLayer):
                    module.anneal_temperature()

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns:
            Dictionary of model configuration
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "task_type": self.task_type,
            "use_gating": self.use_gating,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
        }

    def count_parameters(self) -> int:
        """Count total number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
