from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.evaluate import (
    build_ground_truth,
    coverage,
    diversity_category,
    evaluate_ranking_metrics,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    select_eval_users,
    top_k_from_scores,
)


def test_recall_at_k() -> None:
    rec = ["a", "b", "c", "d"]
    truth = {"a", "d"}
    assert recall_at_k(rec, truth, 2) == 0.5
    assert recall_at_k(rec, truth, 4) == 1.0
    assert recall_at_k(rec, set(), 4) == 0.0


def test_hit_rate_at_k() -> None:
    rec = ["a", "b"]
    assert hit_rate_at_k(rec, {"b"}, 1) == 0.0
    assert hit_rate_at_k(rec, {"b"}, 2) == 1.0


def test_ndcg_at_k_perfectly_ranked() -> None:
    rec = [f"i{i}" for i in range(10)]
    truth = {"i0", "i1", "i2"}
    assert ndcg_at_k(rec, truth, 10) == pytest.approx(1.0)
    # worst case: truth items at positions 4,5,6 -> dcg = sum(1/log2(5..7))
    bad = ["x", "y", "z"] + rec[:10]
    expected = sum(1 / np.log2(n) for n in (5, 6, 7)) / sum(
        1 / np.log2(n) for n in (2, 3, 4)
    )
    assert ndcg_at_k(bad, truth, 10) == pytest.approx(expected)


def test_ndcg_at_k_known_value() -> None:
    # truth = {B, D}; list = [A, B, C, D]; k=4
    rec = ["A", "B", "C", "D"]
    truth = {"B", "D"}
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    assert ndcg_at_k(rec, truth, 4) == pytest.approx(dcg / idcg)


def test_evaluate_ranking_metrics_aggregates() -> None:
    recs = {"u1": ["a", "b", "c"], "u2": ["a", "x", "y"]}
    truth = {"u1": {"a", "b", "z"}, "u2": {"y"}}
    out = evaluate_ranking_metrics(recs, truth, k_values=(2, 3))
    assert out["recall@2"] == pytest.approx(1 / 3)  # u1: 2/3, u2: 0 -> mean
    assert out["hit@3"] == pytest.approx(1.0)  # both hit
    assert out["ndcg@3"] > 0


def test_coverage_fraction() -> None:
    recs = {"u1": ["a", "b"], "u2": ["a", "c"]}
    assert coverage(recs, {"a", "b", "c", "d"}) == pytest.approx(0.75)
    assert coverage({}, {"a", "b"}) == 0.0
    assert coverage(recs, set()) == 0.0


def test_diversity_category() -> None:
    item_cat = {"a": "jacket", "b": "jacket", "c": "shoes", "d": "shoes"}
    recs = {"u1": ["a", "b", "b"], "u2": ["a", "c", "d"]}
    # u1: 1 unique / 3 ; u2: 2 unique / 3
    assert diversity_category(recs, item_cat, 3) == pytest.approx((1 / 3 + 2 / 3) / 2)


def test_top_k_from_scores() -> None:
    assert top_k_from_scores({"a": 0.5, "b": 0.9, "c": 0.1}, 2) == ["b", "a"]


def test_build_ground_truth_excludes_train_items() -> None:
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["a", "b", "a"],
            "event_type": ["view", "view", "view"],
            "timestamp": pd.date_range("2026-01-01", periods=3, tz="UTC"),
        }
    )
    test = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["a", "c", "d"],
            "event_type": ["view", "view", "view"],
            "timestamp": pd.date_range("2026-02-01", periods=3, tz="UTC"),
        }
    )
    truth = build_ground_truth(test, train)
    assert truth == {"u1": {"c"}, "u2": {"d"}}  # "a" seen in train is excluded


def test_build_ground_truth_limited_to_catalog() -> None:
    train = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["a"],
            "event_type": ["view"],
            "timestamp": pd.date_range("2026-01-01", periods=1, tz="UTC"),
        }
    )
    test = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["b", "z"],
            "event_type": ["view", "view"],
            "timestamp": pd.date_range("2026-02-01", periods=2, tz="UTC"),
        }
    )
    # exclusion is per-user: "a" was seen by u1 -> dropped.
    # "z" not in the model catalog -> out of scope, dropped from truth.
    truth = build_ground_truth(test, train, catalog={"a", "b"})
    assert truth == {"u1": {"b"}}


def test_build_ground_truth_excludes_only_users_own_history() -> None:
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_id": ["a", "c"],
            "event_type": ["view", "view"],
            "timestamp": pd.date_range("2026-01-01", periods=2, tz="UTC"),
        }
    )
    test = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_id": ["c", "a"],
            "event_type": ["view", "view"],
            "timestamp": pd.date_range("2026-02-01", periods=2, tz="UTC"),
        }
    )
    # u1 never saw "c" in train -> truth for u1 = {c}; u2 never saw "a" -> {a}
    truth = build_ground_truth(test, train)
    assert truth == {"u1": {"c"}, "u2": {"a"}}


def test_select_eval_users_prefers_rich_history() -> None:
    truth = {"u1": {"a", "b", "c", "d"}, "u2": {"x"}, "u3": {"p", "q"}}
    users = select_eval_users(truth, max_users=2)
    assert users == ["u1", "u3"]


def test_select_eval_users_restrict_filters_cold_start() -> None:
    truth = {"u1": {"a", "b", "c", "d"}, "u2": {"x"}, "u3": {"p", "q"}}
    users = select_eval_users(truth, max_users=10, restrict={"u1", "u3"})
    assert users == ["u1", "u3"]
