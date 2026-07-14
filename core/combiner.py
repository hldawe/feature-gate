"""
P-value combination and multiple testing control (Sections 3.5-3.6).

Implements Fisher's method, Brown's correction, alignment score,
and Benjamini-Hochberg FDR procedure.
"""
import numpy as np
from scipy import stats
from typing import Optional


def fisher_combine(p_values: list[float]) -> float:
    """
    Fisher's method: T = -2 * sum(ln(p_i)), T ~ chi2(2k) under H0.
    """
    valid = [p for p in p_values if p is not None and not np.isnan(p)]
    if not valid:
        return 1.0
    clipped = [max(p, 1e-15) for p in valid]
    T = -2.0 * sum(np.log(p) for p in clipped)
    df = 2 * len(clipped)
    return float(stats.chi2.sf(T, df))


def brown_combine(
    p_values: list[float],
    correlation_matrix: Optional[np.ndarray] = None,
) -> float:
    """
    Brown's method (Section 3.6): Fisher's statistic with dependence-adjusted
    degrees of freedom via moment matching to a scaled chi-squared.

    Falls back to Fisher's method if no correlation matrix is provided.
    """
    valid = [p for p in p_values if p is not None and not np.isnan(p)]
    if not valid:
        return 1.0
    if len(valid) == 1:
        return valid[0]
    if correlation_matrix is None:
        return fisher_combine(p_values)

    clipped = [max(p, 1e-15) for p in valid]
    k = len(clipped)
    T = -2.0 * sum(np.log(p) for p in clipped)

    E_T = 2 * k
    var_T = 4 * k
    for i in range(k):
        for j in range(i + 1, k):
            rho = correlation_matrix[i, j]
            # Kost & McDermott (2002) approximation
            cov_approx = 3.263 * abs(rho) + 0.710 * rho ** 2
            var_T += 2 * cov_approx

    # Match to c * chi2(f): E = c*f, Var = 2*c^2*f
    c = var_T / (2 * E_T)
    f = 2 * E_T ** 2 / var_T

    return float(stats.chi2.sf(T / c, f))


def alignment_score(p_values: list[float], eps: float = 1e-15) -> float:
    """
    Non-inferential ranking heuristic (Section 3.5):
    AlignScore = -sum(log10(p + eps))
    """
    valid = [p for p in p_values if p is not None and not np.isnan(p)]
    if not valid:
        return 0.0
    return float(-sum(np.log10(max(p, eps)) for p in valid))


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """
    Benjamini-Hochberg FDR procedure.
    Returns q-values (adjusted p-values) for each input p-value.
    """
    n = len(p_values)
    if n == 0:
        return []

    p_arr = np.array(p_values, dtype=np.float64)
    valid_mask = ~np.isnan(p_arr)
    valid_p = p_arr[valid_mask]
    m = len(valid_p)

    if m == 0:
        return [float("nan")] * n

    sorted_idx = np.argsort(valid_p)
    sorted_p = valid_p[sorted_idx]

    q_vals = np.empty(m)
    q_vals[sorted_idx[m - 1]] = sorted_p[m - 1]
    for i in range(m - 2, -1, -1):
        rank = i + 1
        adjusted = sorted_p[i] * m / rank
        q_vals[sorted_idx[i]] = min(adjusted, q_vals[sorted_idx[i + 1]])

    q_vals = np.clip(q_vals, 0, 1)

    result = np.full(n, np.nan)
    result[valid_mask] = q_vals
    return result.tolist()
