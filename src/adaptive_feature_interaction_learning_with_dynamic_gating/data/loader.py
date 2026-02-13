"""Data loading utilities for tabular datasets."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_classification,
    make_regression,
    fetch_openml,
)
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    n_samples: int = 10000,
    n_features: int = 50,
    n_informative: int = 30,
    task_type: str = "regression",
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic heterogeneous tabular dataset.

    Args:
        n_samples: Number of samples
        n_features: Total number of features
        n_informative: Number of informative features
        task_type: 'regression' or 'classification'
        n_classes: Number of classes (for classification)
        random_state: Random seed

    Returns:
        Tuple of (features, targets)
    """
    logger.info(
        f"Generating synthetic {task_type} dataset with {n_samples} samples, "
        f"{n_features} features ({n_informative} informative)"
    )

    if task_type == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=10.0,
            random_state=random_state,
        )
    elif task_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=min(5, n_features - n_informative),
            n_classes=n_classes,
            flip_y=0.05,
            class_sep=0.8,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # Add feature interactions to make the problem more complex
    n_interactions = min(10, n_features // 5)
    np.random.seed(random_state)
    for i in range(n_interactions):
        idx1, idx2 = np.random.choice(n_features, size=2, replace=False)
        interaction_feature = X[:, idx1] * X[:, idx2]

        if task_type == "regression":
            y += 0.1 * interaction_feature
        else:
            # For classification, add interaction signal
            interaction_strength = np.percentile(np.abs(interaction_feature), 80)
            mask = np.abs(interaction_feature) > interaction_strength
            if mask.sum() > 0 and n_classes == 2:
                y[mask] = 1 - y[mask]  # Flip some labels based on interaction

    return X, y


def load_openml_dataset(
    dataset_name: str,
    task_type: str = "classification",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from OpenML.

    Args:
        dataset_name: Name or ID of OpenML dataset
        task_type: 'regression' or 'classification'

    Returns:
        Tuple of (features, targets)
    """
    logger.info(f"Loading OpenML dataset: {dataset_name}")

    try:
        # Try to load by name or ID
        try:
            data = fetch_openml(name=dataset_name, version=1, parser='auto')
        except Exception:
            data = fetch_openml(data_id=int(dataset_name), parser='auto')

        X = data.data
        y = data.target

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Handle categorical targets for classification
        if task_type == "classification":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)

        logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        return X, y

    except Exception as e:
        logger.warning(f"Failed to load OpenML dataset {dataset_name}: {e}")
        logger.info("Falling back to synthetic dataset")
        return generate_synthetic_dataset(task_type=task_type)


def load_dataset(
    dataset_name: str = "synthetic",
    task_type: str = "regression",
    data_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from various sources.

    Args:
        dataset_name: Name of dataset ('synthetic', 'openml:<name>', or file path)
        task_type: 'regression' or 'classification'
        data_path: Path to custom data file (CSV)
        **kwargs: Additional arguments for dataset generation

    Returns:
        Tuple of (features, targets)
    """
    # Load from custom file
    if data_path is not None:
        logger.info(f"Loading dataset from file: {data_path}")
        df = pd.read_csv(data_path)

        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if task_type == "classification":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)

        return X, y

    # Load from OpenML
    if dataset_name.startswith("openml:"):
        openml_name = dataset_name.split(":", 1)[1]
        return load_openml_dataset(openml_name, task_type)

    # Generate synthetic
    if dataset_name == "synthetic":
        return generate_synthetic_dataset(task_type=task_type, **kwargs)

    # Try to load as OpenML dataset
    try:
        return load_openml_dataset(dataset_name, task_type)
    except Exception:
        logger.warning(f"Could not load dataset '{dataset_name}', using synthetic")
        return generate_synthetic_dataset(task_type=task_type, **kwargs)


def get_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_state: int = 42,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create train/val/test data loaders.

    Args:
        X: Feature matrix
        y: Target vector
        batch_size: Batch size for training
        val_split: Validation set fraction
        test_split: Test set fraction
        random_state: Random seed for splitting
        num_workers: Number of data loading workers

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_split,
        random_state=random_state,
        stratify=y if len(np.unique(y)) < 20 else None,
    )

    # Second split: separate train and validation
    val_size = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp if len(np.unique(y_temp)) < 20 else None,
    )

    logger.info(
        f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    # Convert to torch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    # Handle classification targets (convert to long)
    if len(np.unique(y)) < 100:  # Likely classification
        y_train_t = y_train_t.long()
        y_val_t = y_val_t.long()
        y_test_t = y_test_t.long()

    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
