"""
Model-free dependence tests (Section 3.4): HSIC and distance correlation.
Also includes lag scan via cross-correlation.
"""
import numpy as np
from typing import Optional
from scipy.spatial.distance import pdist


def _rbf_kernel(X: np.ndarray, sigma: float) -> np.ndarray:
    """RBF (Gaussian) kernel matrix."""
    sq_dists = np.sum(X ** 2, axis=1, keepdims=True) - 2 * X @ X.T + np.sum(X ** 2, axis=1)
    return np.exp(-sq_dists / (2 * sigma ** 2))


def _median_heuristic(X: np.ndarray) -> float:
    """Median heuristic for RBF bandwidth."""
    dists = pdist(X.reshape(-1, 1), metric="euclidean")
    med = float(np.median(dists))
    return max(med, 1e-8)


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix in feature space."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _block_permute(arr: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Block permutation for time-series aware tests."""
    n = len(arr)
    n_blocks = max(1, n // block_size)
    padded_len = n_blocks * block_size
    padded = arr[:padded_len].copy()
    blocks = padded.reshape(n_blocks, block_size)
    rng.shuffle(blocks)
    result = blocks.flatten()
    if padded_len < n:
        result = np.concatenate([result, arr[padded_len:]])
    return result


def hsic_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Compute HSIC statistic with RBF kernels and median heuristic."""
    n = len(x)
    sigma_x = _median_heuristic(x)
    sigma_y = _median_heuristic(y)
    Kx = _center_kernel(_rbf_kernel(x.reshape(-1, 1), sigma_x))
    Ky = _center_kernel(_rbf_kernel(y.reshape(-1, 1), sigma_y))
    return float(np.trace(Kx @ Ky) / ((n - 1) ** 2))


def hsic_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 200,
    block_size: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """HSIC with permutation p-value (block-aware for time series)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) < 10:
        return {"statistic": float("nan"), "p_value": 1.0}

    obs_stat = hsic_statistic(x_clean, y_clean)
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_perm):
        if block_size and block_size > 1:
            y_perm = _block_permute(y_clean, block_size, rng)
        else:
            y_perm = rng.permutation(y_clean)
        null_stat = hsic_statistic(x_clean, y_perm)
        if null_stat >= obs_stat:
            count += 1

    p_value = (1 + count) / (n_perm + 1)
    return {"statistic": float(obs_stat), "p_value": float(p_value)}


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix for 1D array."""
    return np.abs(x[:, None] - x[None, :])


def dcor_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Compute distance correlation statistic."""
    n = len(x)
    if n < 4:
        return 0.0

    A = _pairwise_distances(x)
    B = _pairwise_distances(y)

    A_row = A.mean(axis=1, keepdims=True)
    A_col = A.mean(axis=0, keepdims=True)
    A_grand = A.mean()
    A_centered = A - A_row - A_col + A_grand

    B_row = B.mean(axis=1, keepdims=True)
    B_col = B.mean(axis=0, keepdims=True)
    B_grand = B.mean()
    B_centered = B - B_row - B_col + B_grand

    dcov_xy = np.sqrt(max(0, (A_centered * B_centered).mean()))
    dcov_xx = np.sqrt(max(0, (A_centered * A_centered).mean()))
    dcov_yy = np.sqrt(max(0, (B_centered * B_centered).mean()))

    if dcov_xx * dcov_yy == 0:
        return 0.0
    return float(dcov_xy / np.sqrt(dcov_xx * dcov_yy))


def dcor_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 200,
    block_size: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Distance correlation with permutation p-value."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) < 10:
        return {"statistic": float("nan"), "p_value": 1.0}

    obs_stat = dcor_statistic(x_clean, y_clean)
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_perm):
        if block_size and block_size > 1:
            y_perm = _block_permute(y_clean, block_size, rng)
        else:
            y_perm = rng.permutation(y_clean)
        null_stat = dcor_statistic(x_clean, y_perm)
        if null_stat >= obs_stat:
            count += 1

    p_value = (1 + count) / (n_perm + 1)
    return {"statistic": float(obs_stat), "p_value": float(p_value)}


def lag_scan(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 14,
    positive_only: bool = False,
) -> dict:
    """
    Cross-correlation lag scan (Section 3.4).
    Returns best lag and CCF magnitude.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]
    n = len(x_clean)

    if n < 10 or max_lag < 1:
        return {"best_lag": 0, "best_ccf": 0.0, "ccf_by_lag": {}}

    x_c = x_clean - np.mean(x_clean)
    y_c = y_clean - np.mean(y_clean)
    sx = np.std(x_clean)
    sy = np.std(y_clean)
    if sx == 0 or sy == 0:
        return {"best_lag": 0, "best_ccf": 0.0, "ccf_by_lag": {}}

    lag_min = 0 if positive_only else -max_lag
    lag_max = max_lag

    best_lag = 0
    best_ccf = 0.0
    ccf_by_lag = {}

    for tau in range(lag_min, lag_max + 1):
        if tau >= 0:
            if tau >= n:
                continue
            corr = np.dot(x_c[:n - tau], y_c[tau:]) / (n * sx * sy)
        else:
            abs_tau = abs(tau)
            if abs_tau >= n:
                continue
            corr = np.dot(x_c[abs_tau:], y_c[:n - abs_tau]) / (n * sx * sy)

        ccf_by_lag[tau] = round(float(corr), 6)
        if abs(corr) > abs(best_ccf):
            best_ccf = float(corr)
            best_lag = tau

    return {
        "best_lag": best_lag,
        "best_ccf": best_ccf,
        "ccf_by_lag": ccf_by_lag,
    }
