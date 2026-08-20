"""Online drift detection (Section 17, 19.2) -- PSI / KL over feature distributions.

The online serving path can compare a rolling window of recent requests/events
against a stored baseline (training-time distribution) and emit a scalar drift
score. Thresholds come from config (`monitoring.drift_psi_threshold`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np
import pandas as pd


def _quantile_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile-aligned bin edges derived from the reference distribution."""
    ref = np.asarray(reference, dtype=float)
    if ref.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    if ref.size < 2 or np.all(ref == ref[0]):
        lo = float(ref.min())
        hi = float(ref.max())
        if hi == lo:
            return np.linspace(lo - 1.0, lo + 1.0, n_bins + 1)
        return np.linspace(lo - 1e-6, hi + 1e-6, n_bins + 1)
    edges = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return np.linspace(float(ref.min()), float(ref.max()), n_bins + 1)
    return edges


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=edges)
    total = hist.sum() or 1.0
    p = np.clip(hist / total, 1e-6, None)
    return p / p.sum()


def psi(
    actual: Sequence[float] | pd.Series | np.ndarray,
    expected: Sequence[float] | pd.Series | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index (Section 17). Lower = more stable; >0.1 alerts."""
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    if a.size == 0 or e.size == 0:
        return 0.0
    if a.size == e.size and np.array_equal(a, e):
        return 0.0
    edges = _quantile_edges(e, n_bins)
    pa = _proportions(a, edges)
    pe = _proportions(e, edges)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def kl_divergence(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """KL(P||Q) between reference and current on shared quantile bins."""
    return psi(current, reference, n_bins=n_bins)


def detect_categorical_shift(
    reference: pd.Series, current: pd.Series, top_categories: int = 20
) -> dict:
    """Compare category-frequency distributions; returns per-category deltas + score."""

    def _norm(s: pd.Series) -> dict:
        counts = s.value_counts(normalize=True)
        return counts.head(top_categories).to_dict()

    ref_n = _norm(reference)
    cur_n = _norm(current)
    keys = set(ref_n) | set(cur_n)
    shifts = {k: round(float(cur_n.get(k, 0.0)) - ref_n.get(k, 0.0), 4) for k in keys}
    cats = sorted(keys)
    r = np.array([ref_n.get(c, 0.0) + 1e-6 for c in cats])
    c = np.array([cur_n.get(c, 0.0) + 1e-6 for c in cats])
    r = r / r.sum()
    c = c / c.sum()
    score = float(np.sum((r - c) * np.log(r / c)))
    return {"score": score, "category_deltas": shifts}


def detect_drift(
    reference: pd.DataFrame | dict,
    current: pd.DataFrame | dict,
    numeric_columns: Iterable[str] = (),
    categorical_columns: Iterable[str] = (),
    n_bins: int = 10,
) -> dict:
    """Compute per-feature PSI for numeric columns + category-shift scores.

    Returns {feature: {psi, alert}} for numerics, {feature: {score, category_deltas}}
    for categoricals. ``alert`` is True when psi > 0.1.
    """
    ref_df = reference if isinstance(reference, pd.DataFrame) else pd.DataFrame(reference)
    cur_df = current if isinstance(current, pd.DataFrame) else pd.DataFrame(current)
    out: dict = {}
    for col in numeric_columns:
        if col in ref_df and col in cur_df:
            s = psi(
                cast(pd.Series, cur_df[col]), cast(pd.Series, ref_df[col]), n_bins=n_bins
            )
            out[col] = {"psi": s, "alert": bool(s > 0.1)}
    for col in categorical_columns:
        if col in ref_df and col in cur_df:
            out[col] = detect_categorical_shift(
                cast(pd.Series, ref_df[col]), cast(pd.Series, cur_df[col])
            )
    return out
