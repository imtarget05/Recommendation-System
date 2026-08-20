"""Hugging Face Datasets adapter (Section 5.3).

Loads a HF dataset into the internal schema by column-name mapping.
Kept deliberately small and lazy-imported so the base install stays light.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import BaseDatasetLoader, LoadedData, register

_COLUMN_HINTS = {
    "user_id": ("user", "usr", "customer", "client"),
    "item_id": ("item", "movie", "product", "article", "content"),
    "event": ("event", "interaction", "behaviour", "behavior", "engagement"),
    "rating": ("rating", "score", "stars"),
    "timestamp": ("timestamp", "datetime", "time", "date", "ts"),
    "title": ("title", "name", "product_name", "prod_name"),
    "category": ("category", "genre", "product_type", "department", "type", "context"),
    "tags": ("tag", "topic", "keyword"),
    "description": ("text", "desc", "abstract", "review"),
}


def _pick(cols: list[str], hints: tuple[str, ...]) -> str | None:
    for hint in hints:
        for col in cols:
            if hint in col.lower():
                return col
    return None


@register("huggingface")
class HuggingFaceLoader(BaseDatasetLoader):
    """Generic HF dataset adapter. `version` is the Hugging Face repo id."""

    version = ""

    def __init__(self, version: str | None = None, max_rows: int | None = None) -> None:
        super().__init__(version=version or "")
        if not self.version:
            raise ValueError("huggingface loader requires a repo id as version")
        self.max_rows = max_rows

    def download(self, raw_dir: Path) -> Path:
        # Datasets are served by the HF hub at load time; nothing to stage locally.
        return raw_dir

    def load_raw(self, source: Path) -> dict[str, pd.DataFrame]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "the `datasets` package is required; run `uv sync --extra huggingface`"
            ) from exc
        ds = load_dataset(self.version, split="train")
        if self.max_rows:
            ds = ds.select(range(min(self.max_rows, len(ds))))
        return {"hf": ds.to_pandas()}

    def normalize(self, raw: dict[str, pd.DataFrame]) -> LoadedData:
        df = raw["hf"]
        cols = list(df.columns)
        user_col = _pick(cols, _COLUMN_HINTS["user_id"])
        item_col = _pick(cols, _COLUMN_HINTS["item_id"])
        if not user_col or not item_col:
            raise ValueError(
                "could not detect user/item columns; adapt the dataset schema or "
                "provide a dedicated connector"
            )

        item_ids = pd.unique(df[item_col])
        users = pd.DataFrame({"user_id": pd.unique(df[user_col])})

        interactions = pd.DataFrame(
            {
                "user_id": df[user_col].astype(str),
                "item_id": df[item_col].astype(str),
            }
        )

        event_col = _pick(cols, _COLUMN_HINTS["event"]) or _pick(cols, _COLUMN_HINTS["rating"])
        if event_col and pd.api.types.is_numeric_dtype(df[event_col]):
            thr = load_config_threshold()
            interactions["event_type"] = pd.cut(
                df[event_col], bins=[-1, thr, 1e18], labels=["view", "like"]
            ).astype("string")
        elif event_col:
            interactions["event_type"] = (
                df[event_col]
                .astype(str)
                .str.lower()
                .map(
                    {"purchase": "purchase", "buy": "purchase", "click": "click",
                     "like": "like", "view": "view", "watch": "view"}
                )
                .fillna("view")
            )
        else:
            interactions["event_type"] = "view"

        ts_col = _pick(cols, _COLUMN_HINTS["timestamp"])
        interactions["timestamp"] = (
            pd.to_datetime(df[ts_col], utc=True) if ts_col else pd.Timestamp.now(tz="UTC")
        )

        text_cols = {
            "title": _pick(cols, _COLUMN_HINTS["title"]),
            "category": _pick(cols, _COLUMN_HINTS["category"]),
            "tags": _pick(cols, _COLUMN_HINTS["tags"]),
            "text_description": _pick(cols, _COLUMN_HINTS["description"]),
        }
        item_info = df.groupby(item_col, as_index=False).agg(
            {col: ("first") for col in text_cols.values() if col}
        )
        item_info[item_col] = item_info[item_col].astype(str)
        items = pd.DataFrame({"item_id": item_ids.astype(str)}).merge(
            item_info, on=item_col, how="left"
        )
        for target, source_col in text_cols.items():
            if source_col and source_col in items:
                items[target] = items[source_col].astype(str)
            else:
                items[target] = None
            if source_col and source_col not in (target, item_col):
                items = items.drop(columns=[source_col])

        return LoadedData(
            interactions=interactions,
            items=items,
            users=users,
            meta={
                "source": "huggingface",
                "version": self.version,
                "raw_cols": cols,
            },
        )


def load_config_threshold() -> float:
    from app.config import load_config

    return load_config().connectors.movielens.like_rating_threshold
