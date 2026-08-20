from __future__ import annotations

import pandas as pd
import pytest

from data.preprocessing.split import split_by_user_history, split_interactions_time_based


def _hist(n=100, start="2026-01-01", users=None) -> pd.DataFrame:  # noqa: ANN001, ANN202
    users = users or [f"u{i % 4}" for i in range(n)]
    return pd.DataFrame(
        {
            "user_id": users,
            "item_id": [f"i{i}" for i in range(n)],
            "event_type": ["view"] * n,
            "timestamp": pd.date_range(start, periods=n, freq="D", tz="UTC"),
        }
    )


def test_global_split_is_chronological_and_exhaustive() -> None:
    df = _hist(100)
    result = split_interactions_time_based(df)

    assert len(result.train) + len(result.validation) + len(result.test) == len(df)
    # no overlap, no leak: train max <= validation max <= test max
    assert result.train["timestamp"].max() <= result.validation["timestamp"].min()
    assert result.validation["timestamp"].max() <= result.test["timestamp"].min()
    assert result.thresholds["ratios"] == {"train": 0.7, "validation": 0.1, "test": 0.2}
    assert len(result.train) == 70
    assert len(result.validation) == 10
    assert len(result.test) == 20


def test_global_split_deterministic() -> None:
    df = _hist(50)
    a = split_interactions_time_based(df)
    b = split_interactions_time_based(df)
    assert a.train["item_id"].to_list() == b.train["item_id"].to_list()
    assert a.test["item_id"].to_list() == b.test["item_id"].to_list()


def test_global_split_invalid_ratios_raise() -> None:
    df = _hist(10)
    with pytest.raises(ValueError, match="sum to 1"):
        split_interactions_time_based(df, ratios=(0.5, 0.5, 0.5))


def test_per_user_split_each_user_has_chronological_partition() -> None:
    df = _hist(40, users=[f"u{i % 5}" for i in range(40)])
    result = split_by_user_history(df, min_history=2)

    combined = pd.concat(
        [result.train.assign(s="t"), result.validation.assign(s="v"), result.test.assign(s="e")]
    )
    for user_id, g in combined.groupby("user_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        stages = g["s"].to_list()
        # chronological stage order must be train -> validation -> test
        ranked = [{"t": 0, "v": 1, "e": 2}[s] for s in stages]
        assert ranked == sorted(ranked)
        assert "t" in stages
        if len(g) >= 2:
            assert "e" in stages
    assert (
        len(result.train) + len(result.validation) + len(result.test) == len(df)
    )


def test_per_user_split_short_history_stays_in_train() -> None:
    df = _hist(2, users=["a", "a"])
    result = split_by_user_history(df, min_history=3)
    assert len(result.train) == 2
    assert len(result.test) == 0


def test_timestamp_invariant() -> None:
    # intra-second timestamps must not break per-user ordering
    df = _hist(30)
    df["timestamp"] = pd.date_range("2026-01-01", periods=30, freq="ms", tz="UTC")
    result = split_by_user_history(df, min_history=2)
    assert len(result.train) + len(result.validation) + len(result.test) == 30
