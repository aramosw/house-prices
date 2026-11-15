import numpy as np
import pandas as pd


TOLERANCE = 1e-8
MAX_SWEEPS = 5000


def fit_standardiser(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The mean and standard deviation of each column. A constant column gets a standard deviation of 1, so it stays at 0."""
    values = features.to_numpy(dtype=float)
    mean, std = values.mean(axis=0), values.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def standardise(features: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Subtract the mean and divide by the standard deviation of each column, from the fitting houses."""
    return (features.to_numpy(dtype=float) - mean) / std


def gram(Z: np.ndarray, y_centred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The two quantities that all the fits below need: G = Z'Z / n, and c = Z'y / n.

    Parameters:
    Z (np.ndarray): The standardised columns of the fitting houses.
    y_centred (np.ndarray): The target of the fitting houses, with its mean subtracted.

    Returns:
    tuple: G and c.
    """
    n = len(y_centred)
    return Z.T @ Z / n, Z.T @ y_centred / n


def ridge(G: np.ndarray, c: np.ndarray, alpha: float) -> np.ndarray:
    """The ridge coefficients, (G + alpha I)^-1 c, which exist even when the columns are collinear."""
    return np.linalg.solve(G + alpha * np.eye(len(c)), c)


def soft_threshold(z: float, threshold: float) -> float:
    """Shrink z towards 0 by threshold, and return exactly 0 if |z| is not larger than the threshold."""
    return np.sign(z) * max(abs(z) - threshold, 0.0)


def elastic_net(G: np.ndarray, c: np.ndarray, alpha: float, ratio: float, start: np.ndarray = None) -> np.ndarray:
    """
    The elastic net coefficients, found by coordinate descent. A ratio of 1 is the lasso.

    Each coefficient is updated in turn with the others held fixed, using z_j = c_j - sum over k != j of G_jk beta_k,
    until a whole sweep changes no coefficient by more than TOLERANCE.

    Parameters:
    G (np.ndarray): Z'Z / n.
    c (np.ndarray): Z'y / n.
    alpha (float): The strength of the penalty.
    ratio (float): The share of the penalty that is the lasso (absolute values), between 0 and 1.
    start (np.ndarray): The coefficients to start from. Default is zeros.

    Returns:
    np.ndarray: The coefficients on the standardised scale.
    """
    p = len(c)
    beta = np.zeros(p) if start is None else start.copy()
    for _ in range(MAX_SWEEPS):
        largest_change = 0.0
        for j in range(p):
            if G[j, j] == 0:
                continue
            z = c[j] - G[j] @ beta + G[j, j] * beta[j]
            updated = soft_threshold(z, alpha * ratio) / (G[j, j] + alpha * (1 - ratio))
            largest_change = max(largest_change, abs(updated - beta[j]))
            beta[j] = updated
        if largest_change < TOLERANCE:
            break
    return beta


def alpha_max(c: np.ndarray, ratio: float) -> float:
    """The smallest alpha at which every coefficient of the elastic net is 0."""
    return np.max(np.abs(c)) / ratio


def elastic_net_path(G: np.ndarray, c: np.ndarray, alphas: np.ndarray, ratio: float) -> np.ndarray:
    """
    The elastic net coefficients for each alpha, from the largest to the smallest, each started from the one before it.

    Parameters:
    G (np.ndarray): Z'Z / n.
    c (np.ndarray): Z'y / n.
    alphas (np.ndarray): The values of alpha, in any order.
    ratio (float): The share of the penalty that is the lasso.

    Returns:
    np.ndarray: One row of coefficients for each alpha, in the order of alphas.
    """
    order = np.argsort(-alphas)
    path = np.zeros((len(alphas), len(c)))
    beta = None
    for index in order:
        beta = elastic_net(G, c, alphas[index], ratio, start=beta)
        path[index] = beta
    return path


def to_original_scale(beta: np.ndarray, y_mean: float, mean: np.ndarray, std: np.ndarray) -> tuple[float, np.ndarray]:
    """The intercept and the coefficients in the original units of the columns."""
    coefficients = beta / std
    return y_mean - mean @ coefficients, coefficients
