#!/usr/bin/env python
"""Training script for adaptive feature interaction model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.optim as optim

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
from adaptive_feature_interaction_learning_with_dynamic_gating.training.trainer import (
    AdaptiveTrainer,
)
from adaptive_feature_interaction_learning_with_dynamic_gating.evaluation.analysis import (
    plot_training_curves,
    save_results,
)
from adaptive_feature_interaction_learning_with_dynamic_gating.utils.config import (
    load_config,
    save_config,
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
        description="Train adaptive feature interaction model"
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
        help="Output directory for results",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, or auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging("INFO")

    logger.info("=" * 80)
    logger.info("Adaptive Feature Interaction Learning - Training")
    logger.info("=" * 80)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Set random seeds
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_random_seeds(seed)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    save_config(config, output_dir / "config.yaml")

    try:
        # Load dataset
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

        logger.info(f"Dataset shape: {X.shape}, Task: {task_type}")

        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = TabularPreprocessor(
            scaling_method=data_config.get("scaling_method", "standard"),
            handle_outliers=data_config.get("handle_outliers", True),
        )
        X_processed = preprocessor.fit_transform(X)

        # Create data loaders
        logger.info("Creating data loaders...")
        train_config = config.get("training", {})
        data_loaders = get_data_loaders(
            X_processed,
            y,
            batch_size=train_config.get("batch_size", 32),
            val_split=data_config.get("val_split", 0.15),
            test_split=data_config.get("test_split", 0.15),
            random_state=seed,
        )

        # Initialize model
        logger.info("Initializing model...")
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

        logger.info(f"Model parameters: {model.count_parameters():,}")

        # Initialize optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.get("learning_rate", 0.001),
            weight_decay=train_config.get("weight_decay", 0.01),
        )

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = AdaptiveTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            task_type=task_type,
            use_confidence_loss=train_config.get("use_confidence_loss", True),
            gradient_clip_value=train_config.get("gradient_clip_value", 5.0),
            early_stopping_patience=train_config.get("early_stopping_patience", 15),
            lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
            use_mixed_precision=train_config.get("use_mixed_precision", False),
        )

        # Try to initialize MLflow tracking
        try:
            import mlflow
            mlflow.set_experiment(config.get("experiment_name", "adaptive_feature_interaction"))
            mlflow.start_run()
            mlflow.log_params({
                "model_type": "adaptive_feature_interaction",
                "task_type": task_type,
                "n_features": X_processed.shape[1],
                "n_samples": X.shape[0],
                **model_config,
                **train_config,
            })
            logger.info("MLflow tracking initialized")
            use_mlflow = True
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")
            use_mlflow = False

        # Train model
        logger.info("Starting training...")
        train_results = trainer.train(
            train_loader=data_loaders["train"],
            val_loader=data_loaders["val"],
            num_epochs=train_config.get("num_epochs", 100),
            checkpoint_dir=checkpoint_dir,
        )

        logger.info("Training completed!")
        logger.info(f"Best epoch: {train_results['best_epoch'] + 1}")
        logger.info(f"Best validation loss: {train_results['best_val_loss']:.4f}")
        logger.info(f"Total training time: {train_results['total_time']:.2f}s")

        # Log to MLflow
        if use_mlflow:
            try:
                mlflow.log_metrics({
                    "best_val_loss": train_results["best_val_loss"],
                    "best_epoch": train_results["best_epoch"],
                    "total_time": train_results["total_time"],
                })
                mlflow.log_artifact(str(checkpoint_dir / "best_model.pt"))
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_serializable = {
                k: [float(v) for v in values]
                for k, values in train_results["history"].items()
            }
            json.dump(history_serializable, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

        # Plot training curves
        plot_path = output_dir / "training_curves.png"
        plot_training_curves(train_results["history"], save_path=plot_path)

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.validate(data_loaders["test"])

        logger.info("Test set results:")
        for metric_name, metric_value in test_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

        # Log test metrics to MLflow
        if use_mlflow:
            try:
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            except Exception as e:
                logger.warning(f"Failed to log test metrics to MLflow: {e}")

        # Save final results
        final_results = {
            "config": config,
            "train_results": {
                "best_epoch": train_results["best_epoch"],
                "best_val_loss": float(train_results["best_val_loss"]),
                "total_time": float(train_results["total_time"]),
            },
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "model_info": {
                "num_parameters": model.count_parameters(),
                "sparsity_stats": model.get_sparsity_stats(),
            },
        }

        results_path = output_dir / "results.json"
        save_results(final_results, results_path, format="json")

        # Close MLflow run
        if use_mlflow:
            try:
                mlflow.end_run()
            except Exception:
                pass

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Model saved to {checkpoint_dir / 'best_model.pt'}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
