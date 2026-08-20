"""Monitoring metrics (Sections 19.3-19.5) and LLMOps observability (Section 20).

Pure functions over primitive / numpy inputs so they are dependency-free and
unit-testable without a running recommender.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def catalog_coverage(recommended_ids: Sequence, catalog_size: int) -> float:
    """Fraction of the catalog surfaced by a recommendation list (Section 19.3)."""
    if not recommended_ids or not catalog_size:
        return 0.0
    return len(set(recommended_ids)) / catalog_size


def intra_list_diversity(
    item_ids: Sequence, id_to_index: Mapping, feature_matrix: np.ndarray
) -> float:
    """Mean 1 - cosine similarity across item pairs in one list (19.3 diversity)."""
    fm = np.asarray(feature_matrix, dtype=float)
    idx = [id_to_index[i] for i in item_ids if i in id_to_index]
    if len(idx) < 2:
        return 0.0
    feats = fm[idx]
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    normed = feats / np.clip(norms, 1e-9, None)
    sims = normed @ normed.T
    upper = sims[np.triu_indices(len(idx), k=1)]
    return float(1.0 - upper.mean())


def frequency_map(counts: Mapping) -> dict:
    """Normalise raw occurrence counts into a probability map (Section 19.3)."""
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def list_novelty(item_ids: Sequence, popularity: Mapping) -> float:
    """Average -log2(popularity) over a recommendation list (19.3 / 19.5)."""
    vals = [-np.log2(max(popularity.get(i, 1e-9), 1e-9)) for i in item_ids]
    return float(np.mean(vals)) if vals else 0.0


def personalization(
    user_item_lists: Sequence[Sequence],
    id_to_index: Mapping,
    feature_matrix: np.ndarray,
) -> float:
    """1 - mean pairwise cosine similarity of per-user mean item-feature vectors."""
    if len(user_item_lists) < 2:
        return 0.0
    fm = np.asarray(feature_matrix, dtype=float)
    vecs = []
    for recs in user_item_lists:
        idx = [id_to_index[i] for i in recs if i in id_to_index]
        vecs.append(fm[idx].mean(axis=0) if idx else np.zeros(fm.shape[1]))
    arr = np.vstack(vecs)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    normed = arr / np.clip(norms, 1e-9, None)
    sims = normed @ normed.T
    upper = sims[np.triu_indices(len(user_item_lists), k=1)]
    return float(1.0 - upper.mean()) if upper.size else 0.0


def ctr(clicks: int, impressions: int) -> float:
    """Click-through rate = clicks / impressions (Section 19.3)."""
    return clicks / impressions if impressions else 0.0


def latency_percentiles(latencies_ms: Sequence[float], qs=(50, 95, 99)) -> dict:
    """Latency percentiles for LLM calls (Section 20)."""
    if not latencies_ms:
        return {f"p{q}": 0.0 for q in qs}
    arr = np.asarray(latencies_ms, dtype=float)
    return {f"p{q}": round(float(np.percentile(arr, q)), 6) for q in qs}


def rate(numerator: int, denominator: int) -> float:
    """Safe numerator/denominator ratio."""
    return numerator / denominator if denominator else 0.0


def parse_failure_rate(parser) -> float:
    """Fraction of requests rejected by the schema validator (10A.3)."""
    return rate(getattr(parser, "parse_failures", 0), getattr(parser, "total_requests", 0))


def fallback_rate(parser) -> float:
    """Fraction of requests that degraded to the fallback path (10A.8)."""
    return rate(getattr(parser, "fallbacks", 0), getattr(parser, "total_requests", 0))


def token_efficiency(input_tokens: int, output_tokens: int, budget_tokens: int) -> dict:
    """Token utilization relative to the configured budget (10A.5, 10A.7)."""
    used = input_tokens + output_tokens
    return {
        "tokens_used": used,
        "tokens_remaining": max(budget_tokens - used, 0),
        "utilization": rate(used, budget_tokens),
    }
