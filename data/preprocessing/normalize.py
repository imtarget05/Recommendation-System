"""Normalization utilities: schema validation, weighting, recency, item text."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.config import load_config
from app.schemas.events import EVENT_TYPES, INTERACTION_COLUMNS, normalize_timestamp_col
from app.schemas.items import ITEM_COLUMNS, build_item_text
from app.schemas.users import USER_COLUMNS, USER_OPTIONAL_COLUMNS


def validate_schema(
    df: pd.DataFrame, columns: list[str], name: str, required: list[str]
) -> list[str]:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    fatal = [c for c in required if c not in df.columns or df[c].isna().any()]
    if fatal:
        raise ValueError(f"{name} has missing/NaN values in required columns: {fatal}")
    return missing


def validate_interactions(df: pd.DataFrame, *, drop_unknown_events: bool = True) -> pd.DataFrame:
    """Validate the internal interaction schema; returns a cleaned dataframe."""
    validate_schema(df, INTERACTION_COLUMNS, "interactions", ["user_id", "item_id"])
    df = normalize_timestamp_col(df)
    df = df[["user_id", "item_id", "event_type", "timestamp"]]
    unknown = ~df["event_type"].isin(EVENT_TYPES)
    if unknown.any():
        if drop_unknown_events:
            df = df.loc[~unknown].copy()
        else:
            raise ValueError("interactions contain unsupported event types")
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["event_type"] = df["event_type"].astype(str)
    return df.sort_values("timestamp", kind="stable").drop_duplicates(
        ["user_id", "item_id", "event_type", "timestamp"]
    )


def validate_items(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the internal item schema; fills defaults for missing text fields."""
    validate_schema(df, ITEM_COLUMNS, "items", ["item_id"])
    df = df[ITEM_COLUMNS].copy()
    df["item_id"] = df["item_id"].astype(str)
    for col in ("title", "category", "tags", "text_description"):
        df[col] = df[col].where(df[col].notna(), None)
        if df[col].isna().all():
            df[col] = None
    needs_text = df["text_description"].isna()
    if needs_text.any():
        filled = df.loc[needs_text, ["title", "category", "tags"]].fillna("")
        df.loc[needs_text, "text_description"] = filled.apply(
            lambda r: build_item_text(
                r["title"] or None, r["category"] or None, r["tags"] or None
            ),
            axis=1,
        )
    return df


def validate_users(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the internal user schema (Section 5.4).

    Keeps `user_id` (required) plus any optional columns a loader may add
    (e.g. ``metadata`` for H&M customer attributes). IDs are normalized to str.
    """
    validate_schema(df, USER_COLUMNS, "users", ["user_id"])
    out = df.copy()
    out["user_id"] = out["user_id"].astype(str)
    keep = USER_COLUMNS + [c for c in USER_OPTIONAL_COLUMNS if c in out.columns]
    out = out[keep]
    return out.drop_duplicates("user_id", keep="first")


def event_weight(event_type: str, config=None) -> float:  # noqa: ANN001
    config = config or load_config()
    return getattr(config.interaction_weights, event_type, 1.0)


def apply_event_weights(df: pd.DataFrame, config=None) -> pd.Series:  # noqa: ANN001
    cfg = config or load_config()
    return df["event_type"].map(
        lambda e: getattr(cfg.interaction_weights, e, 1.0)
    ).astype(float)


#: Supported recency time units -> seconds per unit (Section 6.3).
_UNIT_SECONDS = {"days": 86400.0, "hours": 3600.0, "minutes": 60.0, "seconds": 1.0}


def recency_weight(
    timestamps: pd.Series,
    reference: Any = None,
    config: Any = None,
) -> pd.Series:
    """recency_weight = exp(-lambda * age), age in the configured unit (Section 6.3).

    Config knobs:
      recency.lambda        decay rate
      recency.unit          'days' | 'hours' | 'minutes' | 'seconds' (grid seconds)
      recency.max_age_days  optional cap on age (converted to the configured unit);
                            events older than the cap get a floor weight instead of
                            decaying to zero forever.
    """
    cfg = config or load_config()
    unit_seconds = _UNIT_SECONDS.get(cfg.recency.unit)
    if unit_seconds is None:
        raise ValueError(
            f"unsupported recency.unit {cfg.recency.unit!r}; "
            f"expected one of {sorted(_UNIT_SECONDS)}"
        )
    if reference is None:
        reference = timestamps.max()
    age = (reference - timestamps).dt.total_seconds() / unit_seconds
    age = age.clip(lower=0.0)
    if cfg.recency.max_age_days:
        age = age.clip(upper=cfg.recency.max_age_days * 86400.0 / unit_seconds)
    return np.exp(-cfg.recency.lambda_ * age)


def item_interaction_counts(df: pd.DataFrame) -> pd.Series:
    return df["item_id"].value_counts()


def trim_to_max_rows(
    df: pd.DataFrame, max_rows: int | None, mode: str = "strided"
) -> pd.DataFrame:
    """Deterministically cap the interaction table.

    Default mode `strided`: systematic sampling over the (already sorted)
    timeline, so the capped set keeps both temporal spread and catalog coverage.
    Mode `head` keeps the earliest rows (used for small quick subsets).
    """
    if max_rows is None or len(df) <= max_rows:
        return df
    if mode == "head":
        return df.head(max_rows)
    if mode == "strided":
        # Evenly spaced row selection that always includes the first and last
        # observation (keeps the full temporal span, Section 5.5/6.3).
        idx = np.unique(np.linspace(0, len(df) - 1, max_rows).round().astype(int))
        return df.iloc[idx].reset_index(drop=True)
    raise ValueError(f"unsupported trim mode {mode!r}; expected 'head' or 'strided'")


def filter_low_activity_users(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    counts = df["user_id"].value_counts()
    keep = counts[counts >= min_interactions].index
    return df[df["user_id"].isin(keep)]
