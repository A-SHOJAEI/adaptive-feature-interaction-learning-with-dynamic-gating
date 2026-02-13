#!/usr/bin/env python
"""Prediction script for adaptive feature interaction model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch

from adaptive_feature_interaction_learning_with_dynamic_gating.models.model import (
    AdaptiveFeatureInteractionModel,
)
from adaptive_feature_interaction_learning_with_dynamic_gating.data.preprocessing import (
    TabularPreprocessor,
)
from adaptive_feature_interaction_learning_with_dynamic_gating.utils.config import (
    load_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Make predictions with adaptive feature interaction model"
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
        "--input",
        type=str,
        required=True,
        help="Path to input data (CSV file) or single sample as JSON string",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions (optional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, or auto)",
    )
    parser.add_argument(
        "--return-confidence",
        action="store_true",
        help="Return confidence scores with predictions",
    )

    return parser.parse_args()


def load_input_data(input_path: str) -> np.ndarray:
    """Load input data from file or JSON string.

    Args:
        input_path: Path to CSV file or JSON string

    Returns:
        Input data as numpy array

    Raises:
        ValueError: If input format is invalid
    """
    # Try to load as file first
    input_file = Path(input_path)
    if input_file.exists():
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_file)
            return df.values
        elif input_path.endswith('.npy'):
            return np.load(input_file)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")

    # Try to parse as JSON
    try:
        data = json.loads(input_path)
        if isinstance(data, list):
            return np.array([data])
        elif isinstance(data, dict):
            return np.array([list(data.values())])
        else:
            raise ValueError("JSON must be a list or dict")
    except json.JSONDecodeError:
        raise ValueError(
            "Input must be a valid file path or JSON string. "
            "Example: '[1.0, 2.0, 3.0]' or '{\"feat1\": 1.0, \"feat2\": 2.0}'"
        )


def main() -> None:
    """Main prediction function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging("INFO")

    logger.info("=" * 80)
    logger.info("Adaptive Feature Interaction Learning - Prediction")
    logger.info("=" * 80)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    try:
        # Load model checkpoint
        logger.info(f"Loading model from {args.checkpoint}...")
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Initialize model
        model_config = config.get("model", {})
        task_type = config.get("data", {}).get("task_type", "regression")

        # Get input dimension from checkpoint
        state_dict = checkpoint["model_state_dict"]
        input_dim = None
        for key in state_dict.keys():
            if "input_embedding.0.weight" in key:
                input_dim = state_dict[key].shape[1]
                break

        if input_dim is None:
            logger.error("Could not determine input dimension from checkpoint")
            sys.exit(1)

        # Determine output dimension from checkpoint
        for key in state_dict.keys():
            if "output_layer.weight" in key:
                output_dim = state_dict[key].shape[0]
                break

        model = AdaptiveFeatureInteractionModel(
            input_dim=input_dim,
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

        # Load input data
        logger.info("Loading input data...")
        X_input = load_input_data(args.input)
        logger.info(f"Input shape: {X_input.shape}")

        # Validate input dimension
        if X_input.shape[1] != input_dim:
            logger.error(
                f"Input dimension mismatch: expected {input_dim}, got {X_input.shape[1]}"
            )
            sys.exit(1)

        # Note: In production, you should save and load the preprocessor
        # For now, we assume input is already preprocessed or apply simple scaling
        logger.info(
            "Note: Input data should be preprocessed using the same "
            "preprocessing pipeline as training"
        )

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_input).to(device)

        # Make predictions
        logger.info("Making predictions...")
        with torch.no_grad():
            predictions, confidence, gate_info = model(
                X_tensor,
                return_confidence=args.return_confidence,
                return_gate_info=True,
            )

        # Process predictions based on task type
        if task_type == "classification":
            if output_dim > 1:
                # Multi-class
                pred_probs = torch.exp(predictions)  # log_softmax -> probs
                pred_classes = predictions.argmax(dim=-1)
                predictions_np = pred_classes.cpu().numpy()
                probabilities_np = pred_probs.cpu().numpy()
            else:
                # Binary
                pred_probs = torch.sigmoid(predictions)
                pred_classes = (predictions > 0).long()
                predictions_np = pred_classes.cpu().numpy()
                probabilities_np = pred_probs.cpu().numpy()
        else:
            # Regression
            predictions_np = predictions.cpu().numpy()
            probabilities_np = None

        confidence_np = None
        if confidence is not None:
            confidence_np = confidence.cpu().numpy()

        # Display predictions
        logger.info("\n" + "=" * 60)
        logger.info("Predictions:")
        logger.info("=" * 60)

        for i in range(len(predictions_np)):
            pred_value = predictions_np[i]

            if task_type == "classification":
                if output_dim > 1:
                    logger.info(f"Sample {i + 1}: Class {pred_value}")
                    logger.info(f"  Probabilities: {probabilities_np[i]}")
                else:
                    logger.info(f"Sample {i + 1}: Class {pred_value}")
                    logger.info(f"  Probability: {probabilities_np[i][0]:.4f}")
            else:
                logger.info(f"Sample {i + 1}: {pred_value[0]:.4f}")

            if confidence_np is not None:
                logger.info(f"  Confidence: {confidence_np[i][0]:.4f}")

        logger.info("=" * 60)

        # Prepare output results
        results = {
            "predictions": predictions_np.tolist(),
            "task_type": task_type,
        }

        if probabilities_np is not None:
            results["probabilities"] = probabilities_np.tolist()

        if confidence_np is not None:
            results["confidence"] = confidence_np.tolist()

        if gate_info is not None:
            results["sparsity_ratio"] = float(gate_info["sparsity_ratio"].cpu().numpy())

        # Save predictions if output path provided
        if args.output is not None:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"\nPredictions saved to {output_path}")

        logger.info("\n" + "=" * 80)
        logger.info("Prediction completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
