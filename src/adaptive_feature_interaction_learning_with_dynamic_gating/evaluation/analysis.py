"""Analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> None:
    """Plot training curves.

    Args:
        history: Training history dictionary
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot loss curves
    if "train_loss" in history and "val_loss" in history:
        axes[0, 0].plot(history["train_loss"], label="Train Loss")
        axes[0, 0].plot(history["val_loss"], label="Val Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Plot learning rate
    if "learning_rate" in history:
        axes[0, 1].plot(history["learning_rate"])
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Learning Rate")
        axes[0, 1].set_title("Learning Rate Schedule")
        axes[0, 1].set_yscale("log")
        axes[0, 1].grid(True, alpha=0.3)

    # Plot sparsity ratio
    if "sparsity_ratio" in history:
        axes[1, 0].plot(history["sparsity_ratio"])
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Sparsity Ratio")
        axes[1, 0].set_title("Gate Sparsity Over Time")
        axes[1, 0].grid(True, alpha=0.3)

    # Plot gate entropy
    if "gate_entropy" in history:
        axes[1, 1].plot(history["gate_entropy"])
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Gate Entropy")
        axes[1, 1].set_title("Gate Entropy Over Time")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_feature_importance(
    model: nn.Module,
    feature_names: Optional[List[str]] = None,
    n_samples: int = 1000,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Analyze feature importance using gradient-based method.

    Args:
        model: Trained model
        feature_names: List of feature names (optional)
        n_samples: Number of random samples for analysis
        device: Device to run analysis on

    Returns:
        Dictionary mapping feature names to importance scores
    """
    model.eval()
    model = model.to(device)

    # Get input dimension
    input_dim = model.input_dim

    # Generate random input samples
    X_random = torch.randn(n_samples, input_dim, requires_grad=True, device=device)

    # Forward pass
    predictions, _, _ = model(X_random, return_confidence=False)

    # Compute gradients with respect to input
    if predictions.dim() > 1:
        # For multi-output, use sum
        output_sum = predictions.sum()
    else:
        output_sum = predictions.sum()

    output_sum.backward()

    # Compute importance as mean absolute gradient
    importance_scores = X_random.grad.abs().mean(dim=0).cpu().detach().numpy()

    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(input_dim)]

    # Create importance dictionary
    importance_dict = {
        name: float(score)
        for name, score in zip(feature_names, importance_scores)
    }

    # Sort by importance
    importance_dict = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    return importance_dict


def save_results(
    results: Dict[str, Any],
    save_path: Path,
    format: str = "json",
) -> None:
    """Save evaluation results to file.

    Args:
        results: Results dictionary
        save_path: Path to save results
        format: Output format ('json' or 'txt')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {save_path}")
    elif format == "txt":
        with open(save_path, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Results saved to {save_path}")
    else:
        raise ValueError(f"Unknown format: {format}")


def print_results_table(results: Dict[str, Any]) -> None:
    """Print results in a formatted table.

    Args:
        results: Results dictionary
    """
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        elif isinstance(value, (tuple, list)) and len(value) == 3:
            # Confidence interval
            print(f"{key:30s}: {value[1]:.4f} ({value[0]:.4f} - {value[2]:.4f})")
        else:
            print(f"{key:30s}: {value}")

    print("=" * 60 + "\n")


def compare_models(
    results_dict: Dict[str, Dict[str, float]],
    primary_metric: str,
) -> None:
    """Compare multiple model results.

    Args:
        results_dict: Dictionary mapping model names to their metrics
        primary_metric: Primary metric for comparison
    """
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)

    # Print header
    print(f"{'Model':<30s} {primary_metric:<15s} {'Relative Improvement':<20s}")
    print("-" * 80)

    # Get baseline (first model or model with 'baseline' in name)
    baseline_name = None
    for name in results_dict.keys():
        if 'baseline' in name.lower():
            baseline_name = name
            break
    if baseline_name is None:
        baseline_name = list(results_dict.keys())[0]

    baseline_value = results_dict[baseline_name].get(primary_metric, 0.0)

    # Print each model
    for model_name, metrics in results_dict.items():
        value = metrics.get(primary_metric, 0.0)

        if baseline_value != 0:
            improvement = ((value - baseline_value) / baseline_value) * 100
            improvement_str = f"{improvement:+.2f}%"
        else:
            improvement_str = "N/A"

        print(f"{model_name:<30s} {value:<15.4f} {improvement_str:<20s}")

    print("=" * 80 + "\n")
