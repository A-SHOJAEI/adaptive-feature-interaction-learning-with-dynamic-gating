#!/usr/bin/env python
"""Evaluation script for adaptive feature interaction model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from adaptive_feature_interaction_learning_with_dynamic_gating.data.loader import (
    load_dataset,
    get_data_loaders,
)
from adaptive_feature_interaction_learning_with_dynamic_gating.data.preprocessing import (
    TabularPreprocessor,
)
from adaptive_feature_interaction_learning_with_dynamic_gating.models.model import (
    AdaptiveFeatureInteractionModel,
)
from adaptive_feature_interaction_learning_with_dynamic_gating.evaluation.metrics import (
    compute_metrics,
    compute_confidence_intervals,
    compute_relative_improvement,
)
from adaptive_feature_interaction_learning_with_dynamic_gating.evaluation.analysis import (
    analyze_feature_importance,
    print_results_table,
    save_results,
)
from adaptive_feature_interaction_learning_with_dynamic_gating.utils.config import (
    load_config,
    set_random_seeds,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate adaptive feature interaction model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, or auto)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline results for comparison",
    )
    parser.add_argument(
        "--compute-ci",
        action="store_true",
        help="Compute confidence intervals via bootstrap",
    )

    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging("INFO")

    logger.info("=" * 80)
    logger.info("Adaptive Feature Interaction Learning - Evaluation")
    logger.info("=" * 80)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Set random seeds
    seed = config.get("seed", 42)
    set_random_seeds(seed)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load dataset (same as training)
        logger.info("Loading dataset...")
        data_config = config.get("data", {})
        dataset_name = data_config.get("dataset_name", "synthetic")
        task_type = data_config.get("task_type", "regression")

        X, y = load_dataset(
            dataset_name=dataset_name,
            task_type=task_type,
            n_samples=data_config.get("n_samples", 10000),
            n_features=data_config.get("n_features", 50),
            n_informative=data_config.get("n_informative", 30),
            n_classes=data_config.get("n_classes", 2),
            random_state=seed,
        )

        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = TabularPreprocessor(
            scaling_method=data_config.get("scaling_method", "standard"),
            handle_outliers=data_config.get("handle_outliers", True),
        )
        X_processed = preprocessor.fit_transform(X)

        # Create data loaders
        data_loaders = get_data_loaders(
            X_processed,
            y,
            batch_size=config.get("training", {}).get("batch_size", 32),
            val_split=data_config.get("val_split", 0.15),
            test_split=data_config.get("test_split", 0.15),
            random_state=seed,
        )

        # Load model
        logger.info(f"Loading model from {args.checkpoint}...")
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Initialize model
        model_config = config.get("model", {})

        # Determine output dimension
        if task_type == "classification":
            n_classes = len(torch.unique(data_loaders["train"].dataset.tensors[1]))
            output_dim = n_classes if n_classes > 2 else 1
        else:
            output_dim = 1

        model = AdaptiveFeatureInteractionModel(
            input_dim=X_processed.shape[1],
            hidden_dims=model_config.get("hidden_dims", [256, 128, 64]),
            output_dim=output_dim,
            task_type=task_type,
            use_gating=model_config.get("use_gating", True),
            dropout_rate=model_config.get("dropout_rate", 0.2),
            use_batch_norm=model_config.get("use_batch_norm", True),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        logger.info("Model loaded successfully")

        # Evaluate on test set
        logger.info("Evaluating on test set...")

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in data_loaders["test"]:
                data = data.to(device)

                predictions, confidence, gate_info = model(
                    data,
                    return_confidence=True,
                    return_gate_info=True,
                )

                if task_type == "classification":
                    if output_dim > 1:
                        # Multi-class
                        pred_probs = torch.exp(predictions)  # log_softmax -> probs
                        pred_classes = predictions.argmax(dim=-1)
                        all_probabilities.append(pred_probs.cpu().numpy())
                    else:
                        # Binary
                        pred_probs = torch.sigmoid(predictions)
                        pred_classes = (predictions > 0).long()
                        all_probabilities.append(pred_probs.cpu().numpy())

                    all_predictions.append(pred_classes.cpu().numpy())
                else:
                    # Regression
                    all_predictions.append(predictions.cpu().numpy())

                all_targets.append(target.cpu().numpy())

        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        if task_type == "classification":
            all_probabilities = np.concatenate(all_probabilities, axis=0)
        else:
            all_probabilities = None

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = compute_metrics(
            y_true=all_targets,
            y_pred=all_predictions,
            task_type=task_type,
            y_pred_proba=all_probabilities,
        )

        # Print results
        print_results_table(metrics)

        # Compute confidence intervals if requested
        ci_results = None
        if args.compute_ci:
            logger.info("Computing confidence intervals...")
            ci_results = compute_confidence_intervals(
                y_true=all_targets,
                y_pred=all_predictions,
                task_type=task_type,
                n_bootstraps=1000,
                confidence_level=0.95,
                random_state=seed,
            )

            logger.info("\nConfidence Intervals (95%):")
            for metric_name, (lower, mean, upper) in ci_results.items():
                logger.info(f"  {metric_name}: {mean:.4f} ({lower:.4f} - {upper:.4f})")

        # Analyze feature importance
        logger.info("Analyzing feature importance...")
        try:
            feature_importance = analyze_feature_importance(
                model=model,
                n_samples=1000,
                device=device,
            )

            # Save top 10 important features
            top_features = dict(list(feature_importance.items())[:10])
            logger.info("\nTop 10 Most Important Features:")
            for feature_name, importance in top_features.items():
                logger.info(f"  {feature_name}: {importance:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute feature importance: {e}")
            feature_importance = {}

        # Get sparsity statistics
        sparsity_stats = model.get_sparsity_stats()
        logger.info(f"\nSparsity Ratio: {sparsity_stats['sparsity_ratio']:.4f}")
        logger.info(f"Gate Entropy: {sparsity_stats['gate_entropy']:.4f}")

        # Compare with baseline if provided
        relative_improvement = None
        if args.baseline is not None:
            baseline_path = Path(args.baseline)
            if baseline_path.exists():
                with open(baseline_path, 'r') as f:
                    baseline_results = json.load(f)

                baseline_metrics = baseline_results.get("metrics", {})

                # Determine primary metric
                primary_metric = "accuracy" if task_type == "classification" else "rmse"

                relative_improvement = compute_relative_improvement(
                    baseline_metrics=baseline_metrics,
                    model_metrics=metrics,
                    primary_metric=primary_metric,
                )

                logger.info(
                    f"\nRelative improvement over baseline ({primary_metric}): "
                    f"{relative_improvement * 100:.2f}%"
                )

        # Save all results
        eval_results = {
            "metrics": {k: float(v) for k, v in metrics.items()},
            "sparsity_stats": {k: float(v) for k, v in sparsity_stats.items()},
            "feature_importance": {k: float(v) for k, v in list(feature_importance.items())[:20]},
        }

        if ci_results is not None:
            eval_results["confidence_intervals"] = {
                k: [float(lower), float(mean), float(upper)]
                for k, (lower, mean, upper) in ci_results.items()
            }

        if relative_improvement is not None:
            eval_results["relative_improvement"] = float(relative_improvement)

        # Save results as JSON
        results_path = output_dir / "evaluation_results.json"
        save_results(eval_results, results_path, format="json")

        # Save results as text
        results_txt_path = output_dir / "evaluation_results.txt"
        save_results(eval_results, results_txt_path, format="txt")

        logger.info("=" * 80)
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to {output_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
