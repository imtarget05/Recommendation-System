from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import load_config
from data.preprocessing.normalize import (
    apply_event_weights,
    event_weight,
    filter_low_activity_users,
    recency_weight,
    trim_to_max_rows,
    validate_interactions,
    validate_items,
    validate_users,
)


def _df(events, ts=None):  # noqa: ANN001, ANN202
    n = len(events)
    ts = ts or pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "item_id": [f"i{i}" for i in range(n)],
            "event_type": list(events),
            "timestamp": list(ts),
        }
    )


def test_event_weights_follow_config() -> None:
    cfg = load_config()
    assert event_weight("view") == cfg.interaction_weights.view
    assert event_weight("click") == cfg.interaction_weights.click
    assert event_weight("like") == cfg.interaction_weights.like
    assert event_weight("purchase") == cfg.interaction_weights.purchase
    assert event_weight("unknown_event") == 1.0  # safe default


def test_apply_event_weights() -> None:
    df = _df(["view", "click", "like", "purchase"])
    w = apply_event_weights(df)
    assert w.tolist() == [1.0, 2.0, 3.0, 5.0]


def test_recency_weight_decays_and_is_bounded() -> None:
    cfg = load_config()
    ts = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    reference = ts.max()
    w = recency_weight(pd.Series(ts), reference=reference)

    assert w.iloc[0] < w.iloc[-1]  # older -> smaller weight
    assert w.iloc[-1] == 1.0  # most recent -> weight 1
    assert (w <= 1.0).all() and (w > 0).all()
    # analytic check: exp(-lambda * 9 days)
    expected = np.exp(-cfg.recency.lambda_ * 9.0)
    assert w.iloc[0] == pytest.approx(expected)


def test_validate_interactions_cleans_and_sorts() -> None:
    df = _df(["view", "click", "click", "purchase"])
    df.loc[df.index[2], "event_type"] = "hover"  # invalid -> dropped
    out = validate_interactions(df)
    assert "hover" not in out["event_type"].to_list()
    assert "event_type" in out.columns
    assert out["timestamp"].is_monotonic_increasing


def test_validate_interactions_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_interactions(_df(["view"]).drop(columns=["event_type"]))


def test_validate_interactions_null_user_raises() -> None:
    df = _df(["view"])
    df.loc[0, "user_id"] = None
    with pytest.raises(ValueError, match="missing/NaN"):
        validate_interactions(df)


def test_validate_items_fills_missing_text_description() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["i1", "i2"],
            "title": ["Black Jacket", None],
            "category": ["jacket", None],
            "tags": [None, None],
            "text_description": [None, None],
        }
    )
    out = validate_items(items)
    assert out.loc[0, "text_description"] == "Black Jacket jacket"
    assert out.loc[1, "text_description"] == ""


def test_validate_users_dedupes_columns() -> None:
    users = pd.DataFrame({"user_id": ["u1", "u2"]})
    out = validate_users(users)
    assert list(out.columns) == ["user_id"]
    assert out["user_id"].dtype == "string" or str(out["user_id"].dtype).startswith("str")


def test_filter_low_activity_users() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["a", "a", "b"],
            "item_id": ["x", "y", "z"],
            "event_type": ["view", "view", "view"],
            "timestamp": pd.date_range("2026-01-01", periods=3, tz="UTC"),
        }
    )
    out = filter_low_activity_users(df, min_interactions=2)
    assert set(out["user_id"]) == {"a"}


def test_validate_users_preserves_optional_metadata() -> None:
    users = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "metadata": ['{"age": 30}', None],
        }
    )
    out = validate_users(users)
    assert list(out.columns) == ["user_id", "metadata"]
    assert out.loc[0, "metadata"] == '{"age": 30}'

    # duplicate ids collapse, first occurrence wins
    dup = pd.DataFrame({"user_id": ["a", "a"], "metadata": ["x", "y"]})
    out2 = validate_users(dup)
    assert len(out2) == 1
    assert out2.iloc[0]["metadata"] == "x"


def test_validate_interactions_timestamp_is_utc_datetime() -> None:
    from datetime import UTC

    df = _df(["view", "click"])
    out = validate_interactions(df)
    # pandas may expose ns or us resolution depending on version; require
    # a tz-aware datetime dtype (UTC), matching INTERACTION_DTYPES intent.
    assert isinstance(out["timestamp"].dtype, pd.DatetimeTZDtype)
    assert out["timestamp"].dt.tz == UTC


def _recency_cfg(**kwargs):  # noqa: ANN001, ANN202
    from types import SimpleNamespace

    from app.config import RecencyConfig

    params = {"lambda_": 0.1, "unit": "hours", "max_age_days": 0.0}
    params.update(kwargs)
    return SimpleNamespace(recency=RecencyConfig(**params))


def test_recency_weight_honors_configured_unit() -> None:
    ts = pd.date_range("2026-01-01 10:00", periods=3, freq="h", tz="UTC")
    w = recency_weight(pd.Series(ts), reference=ts.max(), config=_recency_cfg(unit="hours"))
    assert w.iloc[2] == pytest.approx(1.0)  # age 0h
    assert w.iloc[1] == pytest.approx(np.exp(-0.1 * 1))  # age 1h
    assert w.iloc[0] == pytest.approx(np.exp(-0.1 * 2))  # age 2h


def test_recency_weight_max_age_caps_old_events() -> None:
    # max_age_days = 1/24 -> cap at 1 hour, so the 2h-old event floors at 1h
    ts = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="UTC")
    cfg = _recency_cfg(unit="hours", max_age_days=1.0 / 24.0)
    w = recency_weight(pd.Series(ts), reference=ts.max(), config=cfg)
    floored = np.exp(-0.1 * 1.0)
    assert w.iloc[0] == pytest.approx(floored)
    assert w.iloc[1] == pytest.approx(floored)
    assert w.iloc[2] == pytest.approx(1.0)


def test_recency_weight_invalid_unit_raises() -> None:
    ts = pd.date_range("2026-01-01", periods=2, tz="UTC")
    with pytest.raises(ValueError, match="unsupported recency.unit"):
        recency_weight(pd.Series(ts), config=_recency_cfg(unit="fortnights"))


def test_trim_to_max_rows_strided_keeps_temporal_spread() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["u"] * 100,
            "item_id": [f"i{i}" for i in range(100)],
            "event_type": ["view"] * 100,
            "timestamp": pd.date_range("2026-01-01", periods=100, freq="D", tz="UTC"),
        }
    )
    out = trim_to_max_rows(df, max_rows=10)
    assert len(out) == 10
    # spread preserved: min and max timestamps retained
    assert out["timestamp"].min() == df["timestamp"].min()
    assert out["timestamp"].max() == df["timestamp"].max()


def test_trim_to_max_rows_head_keeps_earliest() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["u"] * 20,
            "item_id": [f"i{i}" for i in range(20)],
            "event_type": ["view"] * 20,
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC"),
        }
    )
    out = trim_to_max_rows(df, max_rows=5, mode="head")
    assert len(out) == 5
    assert out.iloc[0]["timestamp"] == df.iloc[0]["timestamp"]
