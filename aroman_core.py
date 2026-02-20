from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AromanConfig:
    data_path: Path
    negative_indicators: Sequence[int]
    beta: float = 0.5
    lambda_value: float = 0.5


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def load_decision_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(path, index_col=0)
    else:
        df = pd.read_csv(path, index_col=0)

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    if not all(np.issubdtype(dtype, np.number) for dtype in df.dtypes):
        non_numeric = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
        raise ValueError(f"All criteria must be numeric. Non-numeric columns: {non_numeric}")
    return df.astype(float)


def linear_normalization(matrix: np.ndarray, negative_indicators: set[int]) -> np.ndarray:
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    span = maxs - mins
    positive = _safe_divide(matrix - mins, span)
    negative = _safe_divide(maxs - matrix, span)

    out = positive
    if negative_indicators:
        idx = np.array(sorted(negative_indicators), dtype=int)
        out[:, idx] = negative[:, idx]
    return out


def max_normalization(matrix: np.ndarray, negative_indicators: set[int]) -> np.ndarray:
    maxs = matrix.max(axis=0)
    positive = _safe_divide(matrix, maxs)
    negative = 1 - positive

    out = positive
    if negative_indicators:
        idx = np.array(sorted(negative_indicators), dtype=int)
        out[:, idx] = negative[:, idx]
    return out


def vector_normalization(matrix: np.ndarray, negative_indicators: set[int]) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        if j in negative_indicators:
            transformed = _safe_divide(np.ones_like(col), col)
        else:
            transformed = col
        denom = np.sqrt(np.sum(transformed**2))
        out[:, j] = _safe_divide(transformed, np.full_like(transformed, denom))
    return out


def aggregate_normalizations(norm1: np.ndarray, norm2: np.ndarray, beta: float) -> np.ndarray:
    if not 0 <= beta <= 1:
        raise ValueError("beta must be between 0 and 1")
    return beta * norm1 + (1 - beta) * norm2


def psi_weights(matrix: np.ndarray, negative_indicators: set[int]) -> np.ndarray:
    m, n = matrix.shape
    pv = np.zeros((m, n), dtype=float)
    for j in range(n):
        col = matrix[:, j]
        if j in negative_indicators:
            pv[:, j] = _safe_divide(np.full_like(col, np.min(col)), col)
        else:
            pv[:, j] = _safe_divide(col, np.full_like(col, np.max(col)))

    dpv = np.mean(np.abs(pv - np.mean(pv, axis=0)), axis=0)
    if np.isclose(dpv.sum(), 0):
        return np.full(n, 1 / n)
    return dpv / dpv.sum()


def critic_weights(matrix: np.ndarray, negative_indicators: set[int]) -> np.ndarray:
    normalized = np.zeros_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        if j in negative_indicators:
            normalized[:, j] = _safe_divide(np.full_like(col, np.min(col)), col)
        else:
            normalized[:, j] = _safe_divide(col, np.full_like(col, np.max(col)))

    std = np.std(normalized, axis=0)
    corr = np.corrcoef(normalized.T)
    corr = np.nan_to_num(corr, nan=0.0)
    cj = std * np.sum(1 - corr, axis=1)
    if np.isclose(cj.sum(), 0):
        return np.full(matrix.shape[1], 1 / matrix.shape[1])
    return cj / cj.sum()


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    col_sums = matrix.sum(axis=0)
    p = _safe_divide(matrix, col_sums)
    m = matrix.shape[0]
    k = 1 / np.log(m)

    e = np.zeros(matrix.shape[1], dtype=float)
    for j in range(matrix.shape[1]):
        non_zero = p[:, j] > 0
        e[j] = -k * np.sum(p[non_zero, j] * np.log(p[non_zero, j]))

    d = 1 - e
    if np.isclose(d.sum(), 0):
        return np.full(matrix.shape[1], 1 / matrix.shape[1])
    return d / d.sum()


def aroman_scores(weighted_matrix: np.ndarray, negative_indicators: set[int], lambda_value: float) -> np.ndarray:
    if not 0 <= lambda_value <= 1:
        raise ValueError("lambda_value must be between 0 and 1")
    n = weighted_matrix.shape[1]
    neg_mask = np.isin(np.arange(n), list(negative_indicators))
    l_i = weighted_matrix[:, neg_mask].sum(axis=1)
    a_i = weighted_matrix[:, ~neg_mask].sum(axis=1)
    return l_i + lambda_value * a_i


def copras_scores(weighted_matrix: np.ndarray, negative_indicators: set[int]) -> np.ndarray:
    n = weighted_matrix.shape[1]
    neg_mask = np.isin(np.arange(n), list(negative_indicators))
    pi = weighted_matrix[:, ~neg_mask].sum(axis=1)
    ri = weighted_matrix[:, neg_mask].sum(axis=1)
    m = weighted_matrix.shape[0]
    ri_safe = np.where(ri == 0, np.finfo(float).eps, ri)
    qi = pi + (ri.min() * ri.sum()) / (ri_safe * m)
    return (qi / qi.max()) * 100


def promethee_scores(normalized_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    m, n = normalized_matrix.shape
    preference_indices = np.zeros((m, m, n), dtype=float)
    for k in range(n):
        diff = normalized_matrix[:, [k]] - normalized_matrix[:, k]
        pref = np.clip(diff, 0, 1)
        preference_indices[:, :, k] = pref
    aggregated = np.tensordot(preference_indices, weights, axes=([2], [0]))
    positive = aggregated.mean(axis=1)
    negative = aggregated.mean(axis=0)
    return positive - negative


def ranking_frame(index: Iterable[str], scores: np.ndarray, score_name: str) -> pd.DataFrame:
    index = np.array(list(index))
    order = np.argsort(scores)[::-1]
    ranks = np.arange(1, len(index) + 1)
    return pd.DataFrame(
        {
            "Rank": ranks,
            "Country": index[order],
            score_name: scores[order],
        }
    )
