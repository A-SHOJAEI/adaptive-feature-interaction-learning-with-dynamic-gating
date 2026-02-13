"""Data preprocessing utilities for tabular data."""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class TabularPreprocessor:
    """Preprocessor for tabular data with mixed feature types.

    Handles:
    - Missing value imputation
    - Feature scaling/normalization
    - Categorical encoding
    - Outlier handling

    Args:
        scaling_method: 'standard', 'robust', or None
        handle_outliers: Whether to clip outliers
        outlier_quantiles: Tuple of (lower, upper) quantiles for clipping
        fill_na_strategy: Strategy for missing value imputation
    """

    def __init__(
        self,
        scaling_method: str = "standard",
        handle_outliers: bool = True,
        outlier_quantiles: Tuple[float, float] = (0.01, 0.99),
        fill_na_strategy: str = "median",
    ):
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.outlier_quantiles = outlier_quantiles
        self.fill_na_strategy = fill_na_strategy

        # Initialize components
        self.scaler: Optional[Any] = None
        self.imputer: Optional[SimpleImputer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_bounds: Optional[Dict[int, Tuple[float, float]]] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TabularPreprocessor":
        """Fit preprocessor to training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Optional target vector

        Returns:
            Self
        """
        logger.info("Fitting tabular preprocessor")

        # Handle missing values
        if np.isnan(X).any():
            logger.info(f"Imputing missing values with strategy: {self.fill_na_strategy}")
            self.imputer = SimpleImputer(strategy=self.fill_na_strategy)
            X = self.imputer.fit_transform(X)

        # Compute outlier bounds
        if self.handle_outliers:
            logger.info(
                f"Computing outlier bounds at quantiles: {self.outlier_quantiles}"
            )
            self.feature_bounds = {}
            for i in range(X.shape[1]):
                lower = np.quantile(X[:, i], self.outlier_quantiles[0])
                upper = np.quantile(X[:, i], self.outlier_quantiles[1])
                self.feature_bounds[i] = (lower, upper)

        # Fit scaler
        if self.scaling_method == "standard":
            logger.info("Fitting standard scaler")
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        elif self.scaling_method == "robust":
            logger.info("Fitting robust scaler")
            self.scaler = RobustScaler()
            self.scaler.fit(X)
        elif self.scaling_method is not None:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted preprocessor.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Transformed feature matrix

        Raises:
            ValueError: If preprocessor has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        X_transformed = X.copy()

        # Impute missing values
        if self.imputer is not None:
            X_transformed = self.imputer.transform(X_transformed)

        # Clip outliers
        if self.handle_outliers and self.feature_bounds is not None:
            for i, (lower, upper) in self.feature_bounds.items():
                X_transformed[:, i] = np.clip(X_transformed[:, i], lower, upper)

        # Scale features
        if self.scaler is not None:
            X_transformed = self.scaler.transform(X_transformed)

        return X_transformed

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit preprocessor and transform data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Optional target vector

        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled features back to original scale.

        Args:
            X: Transformed feature matrix

        Returns:
            Features in original scale

        Raises:
            ValueError: If scaler is not available
        """
        if self.scaler is None:
            return X

        return self.scaler.inverse_transform(X)

    def get_feature_stats(self, X: np.ndarray) -> Dict[str, Any]:
        """Compute feature statistics.

        Args:
            X: Feature matrix

        Returns:
            Dictionary of feature statistics
        """
        stats = {
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "missing_ratio": np.isnan(X).sum() / X.size,
            "mean": np.nanmean(X, axis=0).tolist(),
            "std": np.nanstd(X, axis=0).tolist(),
            "min": np.nanmin(X, axis=0).tolist(),
            "max": np.nanmax(X, axis=0).tolist(),
        }

        return stats
