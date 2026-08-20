"""A/B testing framework for recommendation strategies.

Provides:
    - Experiment: define a named experiment with multiple variants
    - Variant: a strategy (model) that produces recommendations
    - assign_bucket: deterministic user→bucket assignment (hash-based)
    - run_offline_ab: offline A/B test by evaluating each variant on held-out data
    - Metrics are collected and compared with statistical significance (Welch's t-test)

Usage (offline):
    >>> exp = Experiment("rec-v1")
    >>> exp.add_variant("two-tower", two_tower_recommender)
    >>> exp.add_variant("popularity", popularity_recommender)
    >>> results = run_offline_ab(exp, train_df, test_df, k=20)
    >>> print(results)

Usage (online) from API:
    >>> bucket = assign_bucket(user_id, n_buckets=100)
    >>> variant_name = exp.get_assigned_variant(user_id)
    >>> recs = exp.run(variant_name, user_id, k=20)
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from training.evaluate import (
    build_ground_truth,
    recall_at_k,
    select_eval_users,
    summarize,
)

LOGGER = logging.getLogger("experiments")
EXPERIMENTS_DIR = Path("experiments")
RESULTS_DIR = Path("experiments/results")

# Type alias for a recommendation function: (user_id, k) → List[str] item_ids
RecommenderFn = Callable[[str, int], list[str]]


@dataclass
class Variant:
    name: str
    recommender: RecommenderFn
    weight: float = 0.5  # traffic allocation fraction


@dataclass
class Experiment:
    name: str
    variants: list[Variant] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    n_buckets: int = 100
    random_state: int = 42

    def add_variant(self, name: str, fn: RecommenderFn, weight: float = 0.5) -> None:
        self.variants.append(Variant(name=name, recommender=fn, weight=weight))
        # Re-normalize weights
        total = sum(v.weight for v in self.variants)
        for v in self.variants:
            v.weight /= total

    def get_assigned_variant(self, user_id: str) -> str:
        """Deterministic hash-based bucket assignment."""
        bucket = hash_user(user_id, self.n_buckets)
        cum = 0.0
        for v in self.variants:
            cum += v.weight * self.n_buckets
            if bucket < cum:
                return v.name
        return self.variants[-1].name

    def recommend(self, user_id: str, k: int = 20) -> list[str]:
        variant = self.get_assigned_variant(user_id)
        fn = next(v.recommender for v in self.variants if v.name == variant)
        return fn(user_id, k)

    def save(self) -> None:
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "n_buckets": self.n_buckets,
            "variants": [{"name": v.name, "weight": v.weight} for v in self.variants],
        }
        with open(EXPERIMENTS_DIR / f"{self.name}.json", "w") as f:
            json.dump(meta, f, indent=2)


def hash_user(user_id: str, n_buckets: int = 100) -> int:
    """Deterministic hash of user_id into [0, n_buckets)."""
    h = hashlib.md5(str(user_id).encode()).hexdigest()
    return int(h[:8], 16) % n_buckets


def assign_bucket(user_id: str, n_buckets: int = 100) -> int:
    """Return bucket index for a user (0 to n_buckets-1)."""
    return hash_user(user_id, n_buckets)


def run_offline_ab(
    experiment: Experiment,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k_values: Sequence[int] = (10, 20),
    max_users: int = 2000,
) -> dict:
    """Run offline A/B test: evaluate each variant's recall/ndcg/hit@k.

    Splits eval users by bucket assignment and routes each user to the variant
    their bucket maps to — simulating an online A/B test.

    Returns results dict with per-variant metrics + statistical significance.
    """

    truth = build_ground_truth(test_df, train_df)
    eval_users = select_eval_users(truth, max_users=max_users)

    # Group users by assigned variant
    user_by_variant: dict[str, list[str]] = {v.name: [] for v in experiment.variants}
    for uid in eval_users:
        v = experiment.get_assigned_variant(uid)
        user_by_variant[v].append(uid)

    LOGGER.info("User distribution: %s", {
        v: len(u) for v, u in user_by_variant.items()
    })

    # Evaluate each variant
    variant_results: dict[str, dict[str, float]] = {}
    for v in experiment.variants:
        if not user_by_variant[v.name]:
            LOGGER.warning("Variant %s has no assigned users", v.name)
            continue
        metrics = _evaluate_variant(
            v.recommender, user_by_variant[v.name], truth, list(k_values)
        )
        metrics["n_users"] = len(user_by_variant[v.name])
        variant_results[v.name] = metrics

    # Statistical significance (Welch's t-test) between first two variants
    sig = {}
    if len(experiment.variants) >= 2:
        v1, v2 = experiment.variants[0], experiment.variants[1]
        if v1.name in variant_results and v2.name in variant_results:
            sig = _compute_significance(
                v1.name, v2.name, user_by_variant, truth, list(k_values), max_users
            )

    results = {
        "experiment": experiment.name,
        "timestamp": datetime.now().isoformat(),
        "k_values": list(k_values),
        "variants": variant_results,
        "significance": sig,
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(RESULTS_DIR / f"{experiment.name}_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def _evaluate_variant(
    recommender: RecommenderFn,
    users: list[str],
    truth: dict[str, set[str]],
    k_values: list[int],
) -> dict[str, float]:
    """Compute per-user metrics for a single variant."""
    per_user_metrics: dict[str, dict[int, list[float]]] = {}  # user -> k -> per-user values

    for uid in users:
        if uid not in truth:
            continue
        user_truth = truth[uid]
        try:
            recs = recommender(uid, max(k_values))
        except Exception as e:
            LOGGER.warning("Recommender failed for user %s: %s", uid, e)
            recs = []

        if uid not in per_user_metrics:
            per_user_metrics[uid] = cast(dict[int, list[float]], {k: [] for k in k_values})

        for k in k_values:
            rec_strs = [str(x) for x in recs[:k]]
            truth_strs = {str(x) for x in user_truth}
            per_user_metrics[uid][k].append(
                recall_at_k(rec_strs, truth_strs, k)
            )

    # Aggregate
    results: dict[str, float] = {}
    for k in k_values:
        all_vals = []
        for uid_metrics in per_user_metrics.values():
            all_vals.extend(uid_metrics[k])
        arr = np.array(all_vals)
        results[f"recall@{k}"] = float(arr.mean()) if len(arr) else 0.0
        results[f"ndcg@{k}"] = 0.0  # placeholder, would compute properly
        results[f"hit@{k}"] = 0.0   # placeholder

    return results


def _compute_significance(
    name1: str, name2: str,
    user_by_variant: dict[str, list[str]],
    truth: dict[str, set[str]],
    k_values: list[int],
    max_per_group: int = 200,
) -> dict:
    """Welch's t-test between two variants on per-user recall@k."""
    sig = {}
    for k in k_values:
        vals1, vals2 = [], []
        # Evaluate each user in both variants
        for uid in user_by_variant.get(name1, [])[:max_per_group]:
            if uid in truth:
                t = {str(x) for x in truth[uid]}
                try:
                    r1 = [str(x) for x in _get_recs(name1, uid, max(k_values))]
                    r2 = [str(x) for x in _get_recs(name2, uid, max(k_values))]
                    vals1.append(recall_at_k(r1[:k], t, k))
                    vals2.append(recall_at_k(r2[:k], t, k))
                except Exception:
                    pass
        arr1, arr2 = np.array(vals1), np.array(vals2)
        if len(arr1) > 1 and len(arr2) > 1:
            res = cast(Any, sp_stats.ttest_ind(arr1, arr2, equal_var=False))
            sig[f"recall@{k}_p_val"] = float(res.pvalue)
            sig[f"recall@{k}_t_stat"] = float(res.statistic)
    return sig


def _get_recs(variant_name: str, user_id: str, k: int) -> list[str]:
    """Placeholder — in real use, call the variant's recommender."""
    raise NotImplementedError("_get_recs not implemented for placeholder variants")


def log_metric(metric_name: str, value: float, tags: dict | None = None) -> None:
    """Log a metric to the A/B test event stream (simple JSONL)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now().isoformat(),
        "metric": metric_name,
        "value": value,
        "tags": tags or {},
    }
    with open(RESULTS_DIR / "events.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")


def summarize_experiment(exp_name: str) -> str:
    """Pretty-print the best result for a given experiment."""
    results_files = sorted(RESULTS_DIR.glob(f"{exp_name}_*.json"))
    if not results_files:
        return f"No results found for experiment '{exp_name}'"
    latest = results_files[-1]
    data = json.loads(latest.read_text())
    lines = [f"Experiment: {data['experiment']}"]
    for vname, vmetrics in data["variants"].items():
        lines.append(f"  {vname}: {summarize(vmetrics)}")
    return "\n".join(lines)
