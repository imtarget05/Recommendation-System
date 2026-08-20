"""Internal normalized user schema (Section 5.4)."""

from __future__ import annotations

from pydantic import BaseModel

#: Required columns for the normalized users table.
USER_COLUMNS = ["user_id"]
#: Optional columns that loaders may add (e.g. H&M customer attributes).
#: `metadata` is stored as a JSON string per user so it survives parquet round-trips.
USER_OPTIONAL_COLUMNS = ["metadata"]
USER_DTYPES = {"user_id": "string"}


class User(BaseModel):
    user_id: str
    metadata: str | dict | None = None
