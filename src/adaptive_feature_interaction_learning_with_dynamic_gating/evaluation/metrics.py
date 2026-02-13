"""Evaluation metrics for model performance assessment."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    average: str = "binary",
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        average: Averaging strategy for multiclass ('binary', 'macro', 'weighted')

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # Add AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                    metrics["auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
            else:
                # Multiclass
                metrics["auc"] = roc_auc_score(
                    y_true,
                    y_pred_proba,
                    multi_class='ovr',
                    average=average,
                )
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")
            metrics["auc"] = 0.0

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)

    metrics = {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    # Add MAPE if no zeros in y_true
    if not np.any(y_true == 0):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics["mape"] = mape

    return metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute metrics based on task type.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        task_type: 'classification' or 'regression'
        y_pred_proba: Predicted probabilities (for classification)

    Returns:
        Dictionary of metrics

    Raises:
        ValueError: If task_type is unknown
    """
    if task_type == "classification":
        # Determine averaging strategy
        n_classes = len(np.unique(y_true))
        average = "binary" if n_classes == 2 else "macro"

        return compute_classification_metrics(
            y_true, y_pred, y_pred_proba, average=average
        )
    elif task_type == "regression":
        return compute_regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def compute_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute confidence intervals via bootstrap.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        task_type: 'classification' or 'regression'
        n_bootstraps: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed

    Returns:
        Dictionary mapping metric name to (lower_bound, mean, upper_bound)
    """
    np.random.seed(random_state)

    n_samples = len(y_true)
    alpha = (1 - confidence_level) / 2

    # Store bootstrap metrics
    bootstrap_metrics = []

    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Compute metrics
        try:
            metrics = compute_metrics(
                y_true_boot,
                y_pred_boot,
                task_type=task_type,
            )
            bootstrap_metrics.append(metrics)
        except Exception:
            # Skip failed bootstrap samples
            continue

    # Compute confidence intervals
    ci_results = {}

    if len(bootstrap_metrics) == 0:
        logger.warning("No valid bootstrap samples, returning empty CI")
        return ci_results

    # Get all metric names
    metric_names = bootstrap_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in bootstrap_metrics]
        lower = np.percentile(values, alpha * 100)
        upper = np.percentile(values, (1 - alpha) * 100)
        mean = np.mean(values)

        ci_results[metric_name] = (lower, mean, upper)

    return ci_results


def compute_relative_improvement(
    baseline_metrics: Dict[str, float],
    model_metrics: Dict[str, float],
    primary_metric: str,
) -> float:
    """Compute relative improvement over baseline.

    Args:
        baseline_metrics: Baseline model metrics
        model_metrics: Current model metrics
        primary_metric: Primary metric for comparison

    Returns:
        Relative improvement (positive means model is better)
    """
    baseline_value = baseline_metrics.get(primary_metric, 0.0)
    model_value = model_metrics.get(primary_metric, 0.0)

    if baseline_value == 0:
        return 0.0

    # For metrics where higher is better (accuracy, f1, r2)
    higher_is_better = primary_metric in ["accuracy", "f1", "precision", "recall", "auc", "r2"]

    if higher_is_better:
        improvement = (model_value - baseline_value) / baseline_value
    else:
        # For metrics where lower is better (mse, mae, rmse)
        improvement = (baseline_value - model_value) / baseline_value

    return improvement
