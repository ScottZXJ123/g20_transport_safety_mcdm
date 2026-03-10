"""Core normalization, weighting, and scoring kernels for the PSI-AROMAN framework.

Implements the methodology described in:

    Zhang et al. (2026). "Machine learning nested MCDM model to enhance
    decision reliability for transport safety engineering."
    Results in Engineering, 29, 108543.

All equation numbers reference the paper above.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "AromanConfig",
    "load_decision_matrix",
    "linear_normalization",
    "max_normalization",
    "vector_normalization",
    "aggregate_normalizations",
    "psi_weights",
    "critic_weights",
    "entropy_weights",
    "aroman_scores",
    "copras_scores",
    "promethee_scores",
    "ranking_frame",
]


@dataclass(frozen=True)
class AromanConfig:
    """Configuration for an AROMAN ranking run.

    Parameters
    ----------
    data_path : Path
        Excel (.xlsx/.xls) or CSV file with countries as index and numeric
        indicator columns.
    negative_indicators : Sequence[int]
        Zero-based column indices that represent cost (lower-is-better) criteria.
    beta : float
        Aggregation weight between the two normalization methods (Eq. 6).
        Default 0.5 gives equal weight.
    lambda_value : float
        Balance between cost and benefit sums in AROMAN scoring (Eq. 14).
        Default 0.5.
    """
    data_path: Path
    negative_indicators: Sequence[int]
    beta: float = 0.5
    lambda_value: float = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division that returns 0 where the denominator is 0."""
    return np.divide(
        numerator, denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_decision_matrix(path: Path) -> pd.DataFrame:
    """Load the decision matrix from *path* (Step 1, Eq. 1).

    The file must have country names/codes as the first column (used as the
    index) and all remaining columns must be numeric indicator values.

    Returns
    -------
    pd.DataFrame
        Float-typed DataFrame with shape (m_countries, n_indicators).
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(path, index_col=0)
    else:
        df = pd.read_csv(path, index_col=0)

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    non_numeric = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric:
        raise ValueError(f"All criteria must be numeric. Non-numeric columns: {non_numeric}")
    return df.astype(float)


# ---------------------------------------------------------------------------
# Normalization  (Step 2)
# ---------------------------------------------------------------------------

def linear_normalization(matrix: np.ndarray, negative_indicators: set[int]) -> np.ndarray:
    """Linear (min-max) normalization (Eqs. 2-3).

    Benefit: x'_ij = (x_ij - min_j) / (max_j - min_j)
    Cost:    x'_ij = (max_j - x_ij) / (max_j - min_j)
    """
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    span = maxs - mins
    benefit = _safe_divide(matrix - mins, span)
    cost = _safe_divide(maxs - matrix, span)

    out = benefit.copy()
    if negative_indicators:
        idx = np.array(sorted(negative_indicators), dtype=int)
        out[:, idx] = cost[:, idx]
    return out


def max_normalization(matrix: np.ndarray, negative_indicators: set[int]) -> np.ndarray:
    """Max normalization (used in initial-sensitivity comparison).

    Benefit: x_ij / max_j
    Cost:    1 - x_ij / max_j
    """
    maxs = matrix.max(axis=0)
    benefit = _safe_divide(matrix, maxs)
    cost = 1.0 - benefit

    out = benefit.copy()
    if negative_indicators:
        idx = np.array(sorted(negative_indicators), dtype=int)
        out[:, idx] = cost[:, idx]
    return out


def vector_normalization(matrix: np.ndarray, negative_indicators: set[int]) -> np.ndarray:
    """Vector (Euclidean) normalization (Eqs. 4-5).

    Benefit: x''_ij = x_ij / sqrt(sum(x_ij^2))
    Cost:    x''_ij = (1/x_ij) / sqrt(sum((1/x_ij)^2))
    """
    out = np.zeros_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        if j in negative_indicators:
            transformed = _safe_divide(np.ones_like(col), col)
        else:
            transformed = col
        denom = np.sqrt(np.sum(transformed ** 2))
        out[:, j] = _safe_divide(transformed, np.full_like(transformed, denom))
    return out


def aggregate_normalizations(norm1: np.ndarray, norm2: np.ndarray, beta: float) -> np.ndarray:
    """Aggregated averaged normalization (Eq. 6).

    x'''_ij = (beta * x'_ij + (1 - beta) * x''_ij) / 2
    """
    if not 0 <= beta <= 1:
        raise ValueError("beta must be between 0 and 1")
    return (beta * norm1 + (1 - beta) * norm2) / 2


# ---------------------------------------------------------------------------
# Weighting  (Step 3)
# ---------------------------------------------------------------------------

def psi_weights(matrix: np.ndarray, negative_indicators: set[int]) -> np.ndarray:
    """Preference Selection Index (PSI) weights (Eqs. 7-11).

    Step 3.1 - preference variation value:
        r_ij = x_min_j / x_ij  (cost)  or  x_ij / x_max_j  (benefit)  [Eqs. 7-8]
        PV_j = sum_i (r_ij - mean(r_j))^2                               [Eq. 9]

    Step 3.2 - deviation of preference value:
        DPV_j = 1 - PV_j                                                [Eq. 10]

    Step 3.3 - weights:
        W_j = DPV_j / sum(DPV_j)                                        [Eq. 11]

    Note: When applied to an already-normalized matrix where cost criteria
    have been inverted, pass ``negative_indicators=set()`` so that all
    columns are treated as benefit.
    """
    m, n = matrix.shape
    r = np.zeros((m, n), dtype=float)
    for j in range(n):
        col = matrix[:, j]
        if j in negative_indicators:
            r[:, j] = _safe_divide(np.full_like(col, col.min()), col)
        else:
            r[:, j] = _safe_divide(col, np.full_like(col, col.max()))

    r_mean = r.mean(axis=0)
    pv = np.sum((r - r_mean) ** 2, axis=0)    # Eq. 9
    dpv = 1.0 - pv                             # Eq. 10
    dpv = np.clip(dpv, 0, None)
    if np.isclose(dpv.sum(), 0):
        return np.full(n, 1.0 / n)
    return dpv / dpv.sum()                     # Eq. 11


def critic_weights(matrix: np.ndarray, negative_indicators: set[int]) -> np.ndarray:
    """CRITIC (CRiteria Importance Through Intercriteria Correlation) weights.

    C_j = sigma_j * sum_k(1 - corr(j,k))
    W_j = C_j / sum(C_j)

    The internal normalization uses preference ratios (min/x for cost,
    x/max for benefit) before computing standard deviations and
    correlations.
    """
    n = matrix.shape[1]
    normalized = np.zeros_like(matrix, dtype=float)
    for j in range(n):
        col = matrix[:, j]
        if j in negative_indicators:
            normalized[:, j] = _safe_divide(np.full_like(col, col.min()), col)
        else:
            normalized[:, j] = _safe_divide(col, np.full_like(col, col.max()))

    std = np.std(normalized, axis=0)
    corr = np.corrcoef(normalized.T)
    corr = np.nan_to_num(corr, nan=0.0)
    cj = std * np.sum(1 - corr, axis=1)
    if np.isclose(cj.sum(), 0):
        return np.full(n, 1.0 / n)
    return cj / cj.sum()


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """Information-entropy weights.

    p_ij = x_ij / sum(x_ij)
    E_j  = -k * sum(p_ij * ln(p_ij)),  k = 1 / ln(m)
    d_j  = 1 - E_j
    W_j  = d_j / sum(d_j)

    All values in *matrix* must be non-negative.
    """
    n = matrix.shape[1]
    col_sums = matrix.sum(axis=0)
    p = _safe_divide(matrix, col_sums)
    m = matrix.shape[0]
    k = 1.0 / np.log(m)

    e = np.zeros(n, dtype=float)
    for j in range(n):
        mask = p[:, j] > 0
        e[j] = -k * np.sum(p[mask, j] * np.log(p[mask, j]))

    d = 1.0 - e
    if np.isclose(d.sum(), 0):
        return np.full(n, 1.0 / n)
    return d / d.sum()


# ---------------------------------------------------------------------------
# Scoring / Aggregation  (Steps 4-5)
# ---------------------------------------------------------------------------

def aroman_scores(weighted_matrix: np.ndarray, negative_indicators: set[int],
                  lambda_value: float) -> np.ndarray:
    """AROMAN aggregation score (Eqs. 13-14).

    L_i = sum of weighted values for cost (negative) criteria   [Eq. 13]
    A_i = sum of weighted values for benefit (positive) criteria [Eq. 13]
    R_i = lambda * L_i + (1 - lambda) * A_i                     [Eq. 14]

    Higher R_i indicates better overall performance.
    """
    if not 0 <= lambda_value <= 1:
        raise ValueError("lambda_value must be between 0 and 1")
    n = weighted_matrix.shape[1]
    neg_mask = np.isin(np.arange(n), list(negative_indicators))
    l_i = weighted_matrix[:, neg_mask].sum(axis=1)
    a_i = weighted_matrix[:, ~neg_mask].sum(axis=1)
    return lambda_value * l_i + (1 - lambda_value) * a_i


def copras_scores(weighted_matrix: np.ndarray, negative_indicators: set[int]) -> np.ndarray:
    """COPRAS (Complex Proportional Assessment) aggregation.

    Q_i = S+_i + sum(S-) / (S-_i * sum(1 / S-_j))
    U_i = (Q_i / Q_max) * 100

    Reference: Chatterjee et al. (2011), Materials & Design.
    """
    n = weighted_matrix.shape[1]
    neg_mask = np.isin(np.arange(n), list(negative_indicators))
    s_plus = weighted_matrix[:, ~neg_mask].sum(axis=1)
    s_minus = weighted_matrix[:, neg_mask].sum(axis=1)
    s_minus_safe = np.where(s_minus == 0, np.finfo(float).eps, s_minus)
    qi = s_plus + s_minus.sum() / (s_minus_safe * np.sum(1.0 / s_minus_safe))
    return (qi / qi.max()) * 100


def promethee_scores(normalized_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """PROMETHEE II net-flow scores.

    Uses a linear preference function (clipped difference).
    phi(a) = phi+(a) - phi-(a), where
        phi+(a) = mean_b pi(a,b),  phi-(a) = mean_b pi(b,a)
        pi(a,b) = sum_k w_k * P_k(a,b)
    """
    m, n = normalized_matrix.shape
    preference_indices = np.zeros((m, m, n), dtype=float)
    for k in range(n):
        diff = normalized_matrix[:, [k]] - normalized_matrix[:, k]
        preference_indices[:, :, k] = np.clip(diff, 0, 1)
    aggregated = np.tensordot(preference_indices, weights, axes=([2], [0]))
    positive_flow = aggregated.mean(axis=1)
    negative_flow = aggregated.mean(axis=0)
    return positive_flow - negative_flow


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def ranking_frame(index: Iterable[str], scores: np.ndarray,
                  score_name: str) -> pd.DataFrame:
    """Build a tidy ranking table sorted by *scores* (descending).

    Returns
    -------
    pd.DataFrame
        Columns: Rank, Country, <score_name>.
    """
    index = np.array(list(index))
    order = np.argsort(scores)[::-1]
    ranks = np.arange(1, len(index) + 1)
    return pd.DataFrame(
        {
            "Rank": ranks,
            "Country": index[order],
            score_name: np.round(scores[order], 6),
        }
    )
