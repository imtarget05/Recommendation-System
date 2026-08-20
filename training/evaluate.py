"""Offline recommendation metrics (Section 16.1): Recall@K, NDCG@K, HitRate@K,
Coverage, Diversity. All helpers are vectorizable and designed for unit testing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

#: A recommendation result: mapping user_id -> ranked list of item_ids
RecsMap = Mapping[str, Sequence[str]]


def recall_at_k(recommended: Sequence[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    hits = len(set(recommended[:k]) & truth)
    return hits / len(truth)


def hit_rate_at_k(recommended: Sequence[str], truth: set[str], k: int) -> float:
    return 1.0 if set(recommended[:k]) & truth else 0.0


def ndcg_at_k(recommended: Sequence[str], truth: set[str], k: int) -> float:
    """DCG with binary relevance, normalized by ideal DCG."""
    if not truth:
        return 0.0
    gains = np.array([1.0 if item in truth else 0.0 for item in recommended[:k]], dtype=float)
    if gains.sum() == 0:
        return 0.0
    dcg = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))
    ideal = float(np.sum(1.0 / np.log2(np.arange(2, min(len(truth), k) + 2))))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_ranking_metrics(
    recommendations: RecsMap, ground_truth: dict[str, set[str]], k_values: Iterable[int] = (10, 20)
) -> dict[str, float]:
    """Aggregate per-user metrics over an evaluation user set."""
    k_values = sorted(k_values)
    tags = [f"{m}@{k}" for m in ("recall", "ndcg", "hit") for k in k_values]
    agg: dict[str, list[float]] = {tag: [] for tag in tags}
    for user, truth in ground_truth.items():
        if not truth:
            continue
        rec = recommendations.get(user, [])
        for k in k_values:
            agg[f"recall@{k}"].append(recall_at_k(rec, truth, k))
            agg[f"ndcg@{k}"].append(ndcg_at_k(rec, truth, k))
            agg[f"hit@{k}"].append(hit_rate_at_k(rec, truth, k))
    return {
        name: (float(np.mean(vals)) if vals else 0.0) for name, vals in agg.items()
    }


def coverage(recommendations: RecsMap, candidate_items: set[str]) -> float:
    """Fraction of candidate items ever recommended."""
    if not recommendations:
        return 0.0
    recommended = {item for items in recommendations.values() for item in items}
    if not candidate_items:
        return 0.0
    return len(recommended & candidate_items) / len(candidate_items)


def diversity_category(
    recommendations: RecsMap, item_to_category: dict[str, str], k: int
) -> float:
    """Mean intra-list category diversity: |unique categories in top-k| / k."""
    values = []
    for items in recommendations.values():
        cats = {item_to_category.get(i) for i in items[:k] if item_to_category.get(i)}
        if not cats:
            continue
        values.append(len(cats) / k)
    return float(np.mean(values)) if values else 0.0


def build_ground_truth(
    test_interactions: pd.DataFrame,
    train: pd.DataFrame,
    catalog: set[str] | None = None,
) -> dict[str, set[str]]:
    """Ground truth per user = items interacted in the eval window.

    Protocol (implicit-feedback, Section 16):
    - Exclude items the user ALREADY interacted with in training (per-user), so
      we measure a model's ability to surface *new* items the user engaged with.
    - Optionally restrict to `catalog` — items the model can possibly output
      (items out of the catalog are unreachable and would distort Recall/NDCG).
    """
    seen_by_user: dict[str, set[str]] = {}
    for user_id, group in train.groupby("user_id"):
        seen_by_user[str(user_id)] = set(group["item_id"])

    truth: dict[str, set[str]] = {}
    for user_id, group in test_interactions.groupby("user_id"):
        items = set(group["item_id"]) - seen_by_user.get(str(user_id), set())
        if catalog is not None:
            items &= catalog
        if items:
            truth[str(user_id)] = items
    return truth


def select_eval_users(
    ground_truth: dict[str, set[str]],
    max_users: int = 2000,
    restrict: set[str] | None = None,
) -> list[str]:
    """Deterministic sample of evaluation users (ranked by history size).

    `restrict` limits evaluation to a known set of users (e.g. users present in
    the training set); users outside it are cold-start and are evaluated separately.
    """
    if restrict is not None:
        ground_truth = {u: t for u, t in ground_truth.items() if u in restrict}
    ranked = sorted(ground_truth.items(), key=lambda kv: -len(kv[1]))
    return [u for u, _ in ranked[:max_users]]


def summarize(results: dict[str, float], rounds: int = 4) -> str:
    rows = [f"{name}: {value:.{rounds}f}" for name, value in sorted(results.items())]
    return " | ".join(rows)


def top_k_from_scores(scores: dict[str, float], k: int) -> list[str]:
    return [item for item, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]


def mean_list_len(items: Iterable[Sequence[Any]]) -> float:
    vals = list(items)
    return float(np.mean([len(x) for x in vals])) if vals else 0.0
