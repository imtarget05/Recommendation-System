"""H&M Personalized Fashion Recommendations loader (Section 5.1).

Requires KAGGLE_USERNAME / KAGGLE_KEY and the optional `kaggle` extra
(`pip install kagglehub`). If credentials are missing, raises a clear error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .base import BaseDatasetLoader, LoadedData, register

TMAP = {
    "transaction_id": "transaction_id",
    "customer_id": "user_id",
    "article_id": "item_id",
    "t_dat": "t_dat",
    "sales_channel_id": None,
    "price": None,
    "product_code": None,
    "product_type_no": None,
    "product_type_name": "category",
    "product_group_name": "product_group_name",
    "graphical_appearance_no": None,
    "graphical_appearance_name": "graphical_appearance_name",
    "colour_group_code": None,
    "colour_group_name": "color",
    "perceived_colour_value_id": None,
    "perceived_colour_value_name": None,
    "perceived_colour_master_id": None,
    "perceived_colour_master_name": None,
    "department_no": None,
    "department_name": "department",
    "index_code": None,
    "index_name": "index_name",
    "index_group_no": None,
    "index_group_name": "index_group_name",
    "section_no": None,
    "section_name": "section",
    "garment_group_no": None,
    "garment_group_name": "garment_group_name",
    "detail_desc": "description",
}


@register("hm")
class HMLoader(BaseDatasetLoader):
    """Loads the H&M dataset from Kaggle and normalizes to the internal schema."""

    version = "v1"
    _kaggle_ref = "michaelacook/h-and-m-personalized-fashion-recommendations"

    def __init__(self, version: str | None = None, max_rows: int | None = None) -> None:
        super().__init__(version=version)
        self.max_rows = max_rows

    def download(self, raw_dir: Path) -> Path:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError(
                "kagglehub is not installed; run `uv sync --extra kaggle`"
            ) from exc
        if not _kaggle_creds_present():
            raise RuntimeError(
                "Kaggle credentials missing; set KAGGLE_USERNAME and KAGGLE_KEY"
            )
        path = Path(kagglehub.dataset_download(self._kaggle_ref, force_download=False))
        print(f"[hm] dataset cached at {path}")
        return path

    def load_raw(self, source: Path) -> dict[str, pd.DataFrame]:
        transactions = pd.read_csv(
            source / "transactions_train.csv", nrows=self.max_rows, engine="c"
        )
        articles = pd.read_csv(source / "articles.csv", engine="c")
        customers = pd.read_csv(source / "customers.csv", engine="c")
        return {"transactions": transactions, "articles": articles, "customers": customers}

    def normalize(self, raw: dict[str, pd.DataFrame]) -> LoadedData:
        tx, articles, customers = raw["transactions"], raw["articles"], raw["customers"]

        interactions = (
            tx[["customer_id", "article_id", "t_dat"]]
            .rename(
                columns={
                    "customer_id": "user_id",
                    "article_id": "item_id",
                    "t_dat": "timestamp",
                }
            )
            .assign(
                user_id=lambda df: df["user_id"].map(lambda x: f"hm_user_{x}"),
                item_id=lambda df: df["item_id"].map(lambda x: f"hm_item_{int(x)}"),
                event_type="purchase",
                timestamp=lambda df: pd.to_datetime(df["timestamp"], utc=True),
            )
            .reindex(
                columns=["user_id", "item_id", "event_type", "timestamp"]
            )  # internal schema order (Section 5.4)
        )

        items = (
            articles.rename(columns={"article_id": "item_id"})
            .assign(item_id=lambda df: df["item_id"].map(lambda x: f"hm_item_{int(x)}"))
            .assign(
                title=lambda df: df["prod_name"].astype(str),
                category=lambda df: df["product_type_name"].astype(str),
                tags=lambda df: (
                    df["product_group_name"].astype(str)
                    + ", "
                    + df["graphical_appearance_name"].astype(str)
                ),
                text_description=lambda df: df.apply(
                    lambda r: " ".join(
                        p
                        for p in (
                            r["prod_name"],
                            r["product_type_name"],
                            r["index_group_name"],
                            r["section_name"],
                            r["graphical_appearance_name"],
                            r["colour_group_name"],
                            r["detail_desc"],
                        )
                        if isinstance(p, str) and p
                    ),
                    axis=1,
                ),
            )
            [["item_id", "title", "category", "tags", "text_description"]]
        )

        users = customers.rename(columns={"customer_id": "user_id"}).assign(
            user_id=lambda df: df["user_id"].map(lambda x: f"hm_user_{x}")
        )
        # Store the remaining customer attributes as a JSON string per row so the
        # optional `metadata` column survives the parquet round-trip (Section 5.4).
        users = users[["user_id"]].assign(
            metadata=users.drop(columns=["user_id"]).apply(
                lambda r: json.dumps(r.to_dict(), ensure_ascii=False, default=str),
                axis=1,
            )
        )

        meta = {
            "source": "kaggle",
            "version": self.version,
            "interaction_events": interactions["event_type"].value_counts().to_dict(),
            "n_items": int(items.shape[0]),
            "n_users": int(interactions["user_id"].nunique()),
        }
        return LoadedData(
            interactions=interactions, items=items, users=users, meta=meta
        )


def _kaggle_creds_present() -> bool:
    import os

    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
