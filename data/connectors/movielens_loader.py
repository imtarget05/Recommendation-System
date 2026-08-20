"""MovieLens loader (Section 5.2) with internal-schema normalization."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from app.config import load_config

from .base import BaseDatasetLoader, LoadedData, register

RATINGS_COLS = {
    "userId": "user_id",
    "movieId": "item_id",
    "rating": "rating",
    "timestamp": "timestamp_unix",
}
MOVIES_COLS = {"movieId": "item_id", "title": "title", "genres": "genres"}
TAGS_COLS = {"movieId": "item_id", "tag": "tag"}


@register("movielens")
class MovieLensLoader(BaseDatasetLoader):
    """Loads ml-latest-small or ml-25m from files.grouplens.org."""

    version = "ml-25m"
    _valid_versions = ("ml-latest-small", "ml-25m")

    def __init__(self, version: str | None = None, max_rows: int | None = None) -> None:
        super().__init__(version=version or self.version)
        if self.version not in self._valid_versions:
            raise ValueError(f"unsupported movielens version {self.version!r}")
        self.max_rows = max_rows

    def download(self, raw_dir: Path) -> Path:
        raw_dir.mkdir(parents=True, exist_ok=True)
        archive = raw_dir / f"{self.version}.zip"
        target = raw_dir / self.version
        if not archive.exists():
            url = f"{load_config().connectors.movielens.base_url}/{self.version}.zip"
            print(f"[movielens] downloading {url} ...")
            import urllib.request

            urllib.request.urlretrieve(str(url), str(archive))  # noqa: S310
        if not (target / "ratings.csv").exists():
            print(f"[movielens] extracting {archive} ...")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(raw_dir)
        return archive

    def load_raw(self, source: Path) -> dict[str, pd.DataFrame]:
        base = source.parent / self.version
        if not (base / "ratings.csv").exists():
            raise FileNotFoundError(f"expected ratings.csv in {base}")
        ratings = pd.read_csv(
            base / "ratings.csv", nrows=self.max_rows, engine="c"
        )
        movies = pd.read_csv(base / "movies.csv", engine="c")
        tags = self._load_tags(base)
        return {"ratings": ratings, "movies": movies, "tags": tags}

    def _load_tags(self, base: Path) -> pd.DataFrame:
        tags = pd.read_csv(base / "tags.csv", usecols=TAGS_COLS.keys(), engine="c")
        return tags.rename(columns=TAGS_COLS)

    def normalize(self, raw: dict[str, pd.DataFrame]) -> LoadedData:
        ratings, movies, tags = raw["ratings"], raw["movies"], raw["tags"]
        if "item_id" not in tags.columns:
            tags = tags.rename(columns=TAGS_COLS)

        top_tags = (
            tags.dropna()
            .assign(item_id=lambda df: df["item_id"].map(lambda x: f"ml_item_{int(x)}"))
            .groupby("item_id")["tag"]
            .agg(lambda s: " ".join(s.value_counts().head(8).index.astype(str)))
            .rename("tags")
        )

        items = (
            movies.rename(columns=MOVIES_COLS)
            .assign(item_id=lambda df: df["item_id"].map(lambda x: f"ml_item_{int(x)}"))
            .merge(top_tags, on="item_id", how="left")
            .assign(
                category=lambda df: df["genres"].str.split("|").str[0],
                tags=lambda df: df["genres"].str.replace("|", ", ", regex=False),
                text_description=lambda df: df.apply(
                    lambda r: " ".join(
                        p
                        for p in (r["title"], r["genres"].replace("|", " ",), r.get("tags") or "")
                        if p
                    ),
                    axis=1,
                ),
            )
            .drop(columns=["genres"])
        )
        items["tags"] = items["tags"].where(items["tags"] != "", None)
        items = items[["item_id", "title", "category", "tags", "text_description"]]

        interactions = (
            ratings.rename(columns=RATINGS_COLS)
            .assign(
                user_id=lambda df: df["user_id"].map(lambda x: f"ml_user_{int(x)}"),
                item_id=lambda df: df["item_id"].map(lambda x: f"ml_item_{int(x)}"),
                timestamp=lambda df: pd.to_datetime(
                    df["timestamp_unix"], unit="s", utc=True
                ),
            )
            .drop(columns=["timestamp_unix"])
        )

        thr = load_config().connectors.movielens.like_rating_threshold
        interactions["event_type"] = pd.cut(
            interactions["rating"], bins=[-1, thr, 5.1], labels=["view", "like"]
        ).astype("string")
        interactions = interactions.drop(columns=["rating"])
        interactions = interactions[["user_id", "item_id", "event_type", "timestamp"]]

        users = pd.DataFrame({"user_id": interactions["user_id"].unique()})

        meta = {
            "source": "movielens",
            "version": self.version,
            "interaction_events": interactions["event_type"].value_counts().to_dict(),
            "n_items": int(items.shape[0]),
            "n_users": int(users.shape[0]),
        }
        return LoadedData(
            interactions=interactions, items=items, users=users, meta=meta
        )
