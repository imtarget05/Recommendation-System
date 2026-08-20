"""Internal normalized interaction schema (Section 5.4)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, field_validator

EVENT_TYPES = ("view", "click", "like", "purchase")

INTERACTION_COLUMNS = ["user_id", "item_id", "event_type", "timestamp"]
INTERACTION_DTYPES = {
    "user_id": "string",
    "item_id": "string",
    "event_type": "string",
    "timestamp": "datetime64[ns, UTC]",
}


class Interaction(BaseModel):
    user_id: str
    item_id: str
    event_type: str
    timestamp: datetime

    @field_validator("event_type")
    @classmethod
    def _valid_event(cls, v: str) -> str:
        if v not in EVENT_TYPES:
            raise ValueError(f"unknown event_type {v!r}; expected one of {EVENT_TYPES}")
        return v

    @field_validator("timestamp")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


def timestamp_from_millis(ms: int) -> datetime:
    """Convert epoch millis to UTC datetime (seconds tolerated too)."""
    if ms > 10**13:  # seconds
        ms = ms * 1000
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


def normalize_timestamp_col(df):  # noqa: ANN001
    """Coerce the timestamp column to UTC datetime64."""
    import pandas as pd

    ts = df["timestamp"]
    if ts.dtype.kind in "iu":  # integer epoch
        ts = pd.to_datetime(ts, unit="s", utc=True)
        huge = ts > pd.Timestamp("3000-01-01", tz="UTC")
        if huge.any():
            ts = pd.to_datetime(df.loc[huge, "timestamp"], unit="ms", utc=True)
        df = df.assign(timestamp=ts)
    else:
        df = df.assign(timestamp=pd.to_datetime(ts, utc=True))
    return df
