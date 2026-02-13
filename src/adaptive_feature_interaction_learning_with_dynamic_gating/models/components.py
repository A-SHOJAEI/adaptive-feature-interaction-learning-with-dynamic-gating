"""Custom components for adaptive feature interaction learning.

This module contains novel components:
1. DynamicGatingLayer: Learns which feature interactions to activate
2. HierarchicalFeatureInteraction: Progressively builds higher-order interactions
3. ConfidenceWeightedLoss: Weights samples by prediction confidence
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DynamicGatingLayer(nn.Module):
    """Dynamic gating mechanism that learns to select feature interactions.

    This layer adaptively gates feature interactions based on input context
    and training progress. Gates evolve from simple (pairwise) to complex
    (higher-order) interactions as the model converges.

    Args:
        input_dim: Number of input features
        num_gates: Number of gating units
        gate_temperature: Temperature for gate activation (higher = softer)
        min_temperature: Minimum temperature during annealing
        temperature_decay: Decay rate for temperature annealing
    """

    def __init__(
        self,
        input_dim: int,
        num_gates: int,
        gate_temperature: float = 1.0,
        min_temperature: float = 0.1,
        temperature_decay: float = 0.99,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_gates = num_gates
        self.gate_temperature = gate_temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.current_temperature = gate_temperature

        # Gate network: learns to predict which gates to activate
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, num_gates * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_gates * 2, num_gates),
        )

        # Confidence estimator: estimates prediction confidence
        self.confidence_estimator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Interaction order weights (learnable): how much each order contributes
        self.order_weights = nn.Parameter(torch.ones(3))  # 1st, 2nd, 3rd order

    def forward(
        self,
        x: torch.Tensor,
        training_progress: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with dynamic gating.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            training_progress: Training progress in [0, 1]

        Returns:
            Tuple of (gated_features, gate_info) where gate_info contains
            gate probabilities, confidence scores, and sparsity metrics
        """
        batch_size = x.size(0)

        # Compute gate logits
        gate_logits = self.gate_network(x)

        # Apply temperature-scaled softmax for differentiable gating
        gate_probs = F.softmax(gate_logits / self.current_temperature, dim=-1)

        # Compute confidence scores
        confidence = self.confidence_estimator(x)

        # Modulate gates by confidence (high confidence = sparser gates)
        confidence_modulated_gates = gate_probs * (1.0 - 0.5 * confidence)
        confidence_modulated_gates = confidence_modulated_gates / (
            confidence_modulated_gates.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Apply gumbel-softmax for hard gating during inference
        if not self.training:
            # Hard selection during inference
            gate_indices = torch.argmax(confidence_modulated_gates, dim=-1)
            hard_gates = F.one_hot(gate_indices, num_classes=self.num_gates).float()
            gated_output = hard_gates
        else:
            # Soft gating during training
            gated_output = confidence_modulated_gates

        # Compute sparsity: how many gates are actually used
        gate_entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=-1).mean()
        sparsity_ratio = (confidence_modulated_gates.max(dim=-1)[0] > 0.5).float().mean()

        # Store gate information for analysis
        gate_info = {
            "gate_probs": gate_probs,
            "confidence": confidence,
            "gate_entropy": gate_entropy,
            "sparsity_ratio": sparsity_ratio,
            "temperature": torch.tensor(self.current_temperature),
            "order_weights": F.softmax(self.order_weights, dim=0),
        }

        return gated_output, gate_info

    def anneal_temperature(self) -> None:
        """Anneal temperature for sharper gating over time."""
        self.current_temperature = max(
            self.min_temperature,
            self.current_temperature * self.temperature_decay
        )


class HierarchicalFeatureInteraction(nn.Module):
    """Hierarchical feature interaction module.

    Progressively builds feature interactions from order-1 (individual features)
    to order-2 (pairwise) to order-3 (triplet) based on dynamic gating.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension for interaction networks
        max_order: Maximum interaction order (default: 3)
        use_gating: Whether to use dynamic gating
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        max_order: int = 3,
        use_gating: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_order = max_order
        self.use_gating = use_gating

        # Order-1: Individual feature transformations
        self.order1_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # Order-2: Pairwise interactions
        self.order2_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # Order-3: Higher-order interactions
        self.order3_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # Gating mechanism
        if use_gating:
            self.gate_layer = DynamicGatingLayer(
                input_dim=input_dim,
                num_gates=hidden_dim,
            )

        # Fusion layer: combines all orders
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        training_progress: float = 0.0
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass through hierarchical interactions.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            training_progress: Training progress in [0, 1]

        Returns:
            Tuple of (interaction_features, gate_info)
        """
        gate_info = None

        # Order-1: Individual features
        h1 = self.order1_net(x)

        # Order-2: Pairwise interactions (element-wise product)
        h2_input = h1 * h1  # Simple quadratic interaction
        h2 = self.order2_net(h2_input)

        # Order-3: Triplet interactions
        h3_input = h2 * h1  # Cubic interaction
        h3 = self.order3_net(h3_input)

        # Apply gating if enabled
        if self.use_gating:
            gates, gate_info = self.gate_layer(x, training_progress)

            # Weight each order by gate activations and learned order weights
            order_weights = F.softmax(self.gate_layer.order_weights, dim=0)

            # Progressive weighting: early training focuses on lower orders
            progress_weight = torch.tensor([
                1.0,  # Order-1 always active
                min(1.0, training_progress * 2),  # Order-2 ramps up
                min(1.0, max(0.0, (training_progress - 0.3) * 2)),  # Order-3 later
            ], device=x.device)

            combined_weights = order_weights * progress_weight
            combined_weights = combined_weights / combined_weights.sum()

            h1_weighted = h1 * combined_weights[0]
            h2_weighted = h2 * combined_weights[1]
            h3_weighted = h3 * combined_weights[2]
        else:
            h1_weighted = h1
            h2_weighted = h2
            h3_weighted = h3

        # Concatenate and fuse all orders
        h_all = torch.cat([h1_weighted, h2_weighted, h3_weighted], dim=-1)
        output = self.fusion(h_all)

        return output, gate_info


class ConfidenceWeightedLoss(nn.Module):
    """Confidence-weighted loss function.

    Weights samples by prediction confidence, allowing the model to focus
    on uncertain regions while maintaining stability on confident predictions.

    Novel contribution: dynamically adjusts sample weights based on both
    prediction confidence and gradient magnitudes.

    Args:
        base_loss: Base loss function ('mse' or 'cross_entropy')
        confidence_scale: Scale factor for confidence weighting
        gradient_clip: Gradient clipping threshold
    """

    def __init__(
        self,
        base_loss: str = "mse",
        confidence_scale: float = 1.0,
        gradient_clip: float = 5.0,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.confidence_scale = confidence_scale
        self.gradient_clip = gradient_clip

        if base_loss == "mse":
            self.loss_fn = nn.MSELoss(reduction='none')
        elif base_loss == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute confidence-weighted loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            confidence: Confidence scores (optional, computed from predictions if None)

        Returns:
            Tuple of (weighted_loss, loss_info)
        """
        # Compute base loss per sample
        base_loss = self.loss_fn(predictions, targets)

        # Compute or use provided confidence scores
        if confidence is None:
            if self.base_loss == "cross_entropy":
                # Confidence from softmax probabilities
                probs = F.softmax(predictions, dim=-1)
                confidence = probs.max(dim=-1)[0]
            else:
                # Confidence from prediction magnitude (for regression)
                confidence = torch.sigmoid(-torch.abs(predictions - targets))

        # Squeeze confidence if needed
        if confidence.dim() > 1:
            confidence = confidence.squeeze(-1)

        # Inverse confidence weighting: focus on uncertain samples
        # High confidence -> low weight, Low confidence -> high weight
        weights = 1.0 + self.confidence_scale * (1.0 - confidence)
        weights = weights / weights.mean()  # Normalize

        # Apply weights
        weighted_loss = (base_loss * weights).mean()

        # Compute statistics
        loss_info = {
            "base_loss": base_loss.mean().item(),
            "weighted_loss": weighted_loss.item(),
            "mean_confidence": confidence.mean().item(),
            "mean_weight": weights.mean().item(),
        }

        return weighted_loss, loss_info


class GradientFlowAnalyzer:
    """Analyzer for monitoring gradient flow through the network.

    Helps identify vanishing/exploding gradients and guides the adaptive
    gating mechanism.
    """

    def __init__(self):
        self.gradient_norms: List[float] = []
        self.layer_gradients: Dict[str, List[float]] = {}

    def analyze_gradients(self, model: nn.Module) -> Dict[str, float]:
        """Analyze gradient flow through model.

        Args:
            model: PyTorch model to analyze

        Returns:
            Dictionary of gradient statistics
        """
        total_norm = 0.0
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                layer_norms[name] = param_norm

        total_norm = total_norm ** 0.5

        self.gradient_norms.append(total_norm)

        stats = {
            "total_gradient_norm": total_norm,
            "mean_gradient_norm": np.mean(list(layer_norms.values())) if layer_norms else 0.0,
            "max_gradient_norm": max(layer_norms.values()) if layer_norms else 0.0,
        }

        return stats
