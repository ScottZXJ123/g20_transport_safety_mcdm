"""Core normalization, weighting, and scoring kernels for the PSI-AROMAN framework.

Implements the methodology described in:

    Zhang, X., Zhang, N. (A.), Li, J., Li, Q., Liu, X., Cao, C. (L.),
    Mao, H., Yan, R., Qi, Y., Yang, X. (C.), Li, J., Zhou, A. K.,
    Yan, X., Feng, H., & Chen, F. (2026). Machine learning nested MCDM
    model to enhance decision reliability for transport safety engineering.
    Results in Engineering, 29, 108543.
    https://doi.org/10.1016/j.rineng.2025.108543

Each algorithm implementation follows its canonical definition from the
original literature. Equation numbers in docstrings reference the paper
above; algorithm references cite the original sources.
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

    Reference
    ---------
    Weitendorf, D. (1976). Beitrag zur Optimierung der raumlichen
    Struktur eines Gebaudes. Dissertation, Hochschule fur Architektur
    und Bauwesen, Weimar.
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

    Reference
    ---------
    Stopp, F. (1975). Variantenvergleich durch Matrixspiele. Wiss. Z.
    Tech. Hochsch. Leipzig, 3, 117-118.
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
    """Vector (Euclidean) normalization.

    Benefit: r_ij = x_ij / sqrt(sum_i x_ij^2)
    Cost:    r_ij = 1 - x_ij / sqrt(sum_i x_ij^2)

    This follows the standard formulation where cost criteria are
    inverted by subtracting the benefit-normalized value from 1.

    Reference
    ---------
    Hwang, C.-L., & Yoon, K. (1981). Multiple Attribute Decision Making:
    Methods and Applications. Springer-Verlag, Berlin.
    """
    norms = np.sqrt(np.sum(matrix ** 2, axis=0))
    benefit = _safe_divide(matrix, norms)

    out = benefit.copy()
    if negative_indicators:
        idx = np.array(sorted(negative_indicators), dtype=int)
        out[:, idx] = 1.0 - benefit[:, idx]
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

    Reference
    ---------
    Maniya, K., & Bhatt, M. G. (2010). A selection of material using a
    novel type decision-making method: Preference selection index method.
    Materials & Design, 31(4), 1785-1789.
    https://doi.org/10.1016/j.matdes.2009.11.020
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

    Standard formulation:
      1. Min-max normalize the decision matrix.
      2. sigma_j = std of normalized column j.
      3. r_jk   = Pearson correlation between columns j and k.
      4. C_j    = sigma_j * sum_k(1 - r_jk)
      5. W_j    = C_j / sum(C_j)

    Reference
    ---------
    Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining
    objective weights in multiple criteria problems: The CRITIC method.
    Computers & Operations Research, 22(7), 763-770.
    https://doi.org/10.1016/0305-0548(94)00059-H
    """
    n = matrix.shape[1]
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    span = maxs - mins

    # Standard CRITIC uses min-max normalization (Diakoulaki et al., 1995)
    benefit = _safe_divide(matrix - mins, span)
    cost = _safe_divide(maxs - matrix, span)
    normalized = benefit.copy()
    if negative_indicators:
        idx = np.array(sorted(negative_indicators), dtype=int)
        normalized[:, idx] = cost[:, idx]

    std = np.std(normalized, axis=0, ddof=0)
    corr = np.corrcoef(normalized.T)
    corr = np.nan_to_num(corr, nan=0.0)
    cj = std * np.sum(1 - corr, axis=1)
    if np.isclose(cj.sum(), 0):
        return np.full(n, 1.0 / n)
    return cj / cj.sum()


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """Shannon entropy weights.

    p_ij = x_ij / sum_i(x_ij)
    E_j  = -k * sum_i(p_ij * ln(p_ij)),  k = 1 / ln(m)
    d_j  = 1 - E_j
    W_j  = d_j / sum(d_j)

    All values in *matrix* must be non-negative.

    References
    ----------
    Shannon, C. E. (1948). A mathematical theory of communication. The
    Bell System Technical Journal, 27(3), 379-423.
    https://doi.org/10.1002/j.1538-7305.1948.tb01338.x

    Zeleny, M. (1982). Multiple Criteria Decision Making. McGraw-Hill,
    New York.
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
    """AROMAN aggregation score.

    L_i = sum of weighted values for cost (negative) criteria
    A_i = sum of weighted values for benefit (positive) criteria
    R_i = L_i^lambda + A_i^(1 - lambda)

    With lambda = 0.5 this becomes R_i = sqrt(L_i) + sqrt(A_i).
    Higher R_i indicates better overall performance.

    Reference
    ---------
    Boskovic, S., Svadlenka, L., Jovkovic, S., Dobrodolac, M.,
    Simic, V., & Bacanin, N. (2023). An alternative ranking order
    method accounting for two-step normalization (AROMAN) - A case
    study of the e-commerce development in the European Union.
    IEEE Access, 11, 129820-129838.
    https://doi.org/10.1109/ACCESS.2023.3332357
    """
    if not 0 <= lambda_value <= 1:
        raise ValueError("lambda_value must be between 0 and 1")
    n = weighted_matrix.shape[1]
    neg_mask = np.isin(np.arange(n), list(negative_indicators))
    l_i = weighted_matrix[:, neg_mask].sum(axis=1)
    a_i = weighted_matrix[:, ~neg_mask].sum(axis=1)
    return np.power(l_i, lambda_value) + np.power(a_i, 1 - lambda_value)


def copras_scores(weighted_matrix: np.ndarray, negative_indicators: set[int]) -> np.ndarray:
    """COPRAS (Complex Proportional Assessment) aggregation.

    Q_i = S+_i + sum(S-) / (S-_i * sum(1 / S-_j))
    U_i = (Q_i / Q_max) * 100

    Reference
    ---------
    Zavadskas, E. K., Kaklauskas, A., & Sarka, V. (1994). The new method
    of multicriteria complex proportional assessment of projects.
    Technological and Economic Development of Economy, 1(3), 131-139.
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

    Uses the V-shape (Type III) preference function with threshold p=1:
        P_k(a, b) = clip(f_k(a) - f_k(b), 0, 1)

    Aggregated preference:
        pi(a, b) = sum_k w_k * P_k(a, b)

    Flows (standard divisor m-1, excluding self-comparison):
        phi+(a) = 1/(m-1) * sum_{b != a} pi(a, b)
        phi-(a) = 1/(m-1) * sum_{b != a} pi(b, a)
        phi(a)  = phi+(a) - phi-(a)

    Reference
    ---------
    Brans, J.-P., Vincke, Ph., & Mareschal, B. (1986). How to select
    and how to rank projects: The PROMETHEE method. European Journal of
    Operational Research, 24(2), 228-238.
    https://doi.org/10.1016/0377-2217(86)90044-5
    """
    m, n = normalized_matrix.shape
    preference_indices = np.zeros((m, m, n), dtype=float)
    for k in range(n):
        diff = normalized_matrix[:, [k]] - normalized_matrix[:, k]
        preference_indices[:, :, k] = np.clip(diff, 0, 1)
    aggregated = np.tensordot(preference_indices, weights, axes=([2], [0]))
    # Standard PROMETHEE II divides by (m - 1), excluding self-comparison
    divisor = max(m - 1, 1)
    positive_flow = aggregated.sum(axis=1) / divisor
    negative_flow = aggregated.sum(axis=0) / divisor
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
