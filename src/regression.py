import numpy as np
import pandas as pd


def design_matrix(features: pd.DataFrame) -> np.ndarray:
    """The feature columns, with a column of ones in front for the intercept."""
    return np.column_stack([np.ones(len(features)), features.to_numpy(dtype=float)])


def fit_ols(features: pd.DataFrame, y) -> np.ndarray:
    """
    Fit least squares with an intercept. lstsq is the stable form of the normal equations.

    Parameters:
    features (pd.DataFrame): The feature columns.
    y (array-like): The target.

    Returns:
    np.ndarray: The coefficients, with the intercept first.
    """
    return np.linalg.lstsq(design_matrix(features), np.asarray(y, dtype=float), rcond=None)[0]


def predict(features: pd.DataFrame, coefficients: np.ndarray) -> np.ndarray:
    """The predicted target of each house."""
    return design_matrix(features) @ coefficients


def t_values(features: pd.DataFrame, y) -> pd.Series:
    """
    The t-statistic of each coefficient, with the intercept first.

    The standard errors are sigma squared times the diagonal of (X'X)^-1, which the SVD of X gives as V S^-2 V'.

    Parameters:
    features (pd.DataFrame): The feature columns.
    y (array-like): The target.

    Returns:
    pd.Series: The t-statistic of each coefficient, indexed by column name.
    """
    X = design_matrix(features)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ coefficients
    sigma_squared = residuals @ residuals / (n - k)
    _, singular_values, right_vectors = np.linalg.svd(X, full_matrices=False)
    standard_errors = np.sqrt(sigma_squared * np.sum((right_vectors.T / singular_values) ** 2, axis=1))
    return pd.Series(coefficients / standard_errors, index=["Intercept", *features.columns])
