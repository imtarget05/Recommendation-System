from __future__ import annotations

import json
import shutil
from typing import cast

import pandas as pd
import pytest

from data.connectors.base import available_loaders, get_loader
from data.connectors.movielens_loader import MovieLensLoader
from data.preprocessing.normalize import validate_users
from tests.unit.fakes import FakeMovielensLoader


def test_loader_registry_contains_core_names() -> None:
    names = available_loaders()
    assert "movielens" in names
    assert "hm" in names
    assert "huggingface" in names


def test_get_unknown_loader_raises() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        get_loader("nonexistent")


def test_get_loader_returns_movielens() -> None:
    loader = get_loader("movielens", version="ml-latest-small")
    assert isinstance(loader, MovieLensLoader)
    assert loader.version == "ml-latest-small"


def test_movielens_unsupported_version_raises() -> None:
    with pytest.raises(ValueError, match="unsupported movielens version"):
        MovieLensLoader(version="ml-wrong")


def test_movielens_normalize(sample_movielens_data, tmp_path) -> None:  # noqa: ANN001
    loader = FakeMovielensLoader(tmp_path)
    raw = {
        "ratings": pd.read_csv(sample_movielens_data["ratings"]),
        "movies": pd.read_csv(sample_movielens_data["movies"]),
        "tags": pd.read_csv(sample_movielens_data["tags"]),
    }
    data = loader.normalize(raw)

    assert list(data.interactions.columns) == ["user_id", "item_id", "event_type", "timestamp"]
    assert cast(pd.Series, data.interactions["event_type"]).isin(["view", "like"]).all()

    # rating 4.5 and 5.0 -> like; 2.0 and 3.0 -> view (threshold 4.0)
    ev = data.interactions.set_index(["user_id", "item_id"])["event_type"]
    assert ev[("ml_user_1", "ml_item_10")] == "like"
    assert ev[("ml_user_1", "ml_item_11")] == "view"
    assert ev[("ml_user_3", "ml_item_12")] == "view"

    # id prefixing
    assert data.interactions["user_id"].str.startswith("ml_user_").all()
    assert data.interactions["item_id"].str.startswith("ml_item_").all()

    # timestamps parsed as UTC datetimes (epoch seconds, not millis)
    assert pd.api.types.is_datetime64_any_dtype(data.interactions["timestamp"])
    assert data.interactions["timestamp"].min().year == 2001  # fixture 1000000000s

    # item schema with title/category/tags/text
    assert list(data.items.columns) == ["item_id", "title", "category", "tags", "text_description"]
    toy = data.items[data.items["item_id"] == "ml_item_10"].iloc[0]
    assert toy["title"] == "Toy Story (1995)"
    assert toy["category"] == "Adventure"
    assert "Animation" in toy["tags"]
    assert "Sci-Fi" not in toy["tags"]  # no tag invented

    # user table
    assert set(data.users["user_id"]) == {"ml_user_1", "ml_user_2", "ml_user_3"}

    assert data.meta["source"] == "movielens"


def test_movielens_load_respects_max_rows(sample_movielens_data, tmp_path) -> None:  # noqa: ANN001
    # lay out an on-disk MovieLens layout: raw/ml-latest-small/{ratings,movies,tags}.csv
    version_dir = tmp_path / "ml-latest-small"
    version_dir.mkdir()
    for f in ("ratings", "movies", "tags"):
        shutil.copy(sample_movielens_data[f], version_dir / f"{f}.csv")
    (tmp_path / "ml-latest-small.zip").write_text("")  # fake archive for download()

    loader = FakeMovielensLoader(tmp_path, version="ml-latest-small")
    loader.max_rows = 3
    data = loader.load(tmp_path)

    assert len(data.interactions) == 3
    assert data.source_path == tmp_path / "ml-latest-small.zip"


def test_hm_normalize_produces_internal_schema(sample_hm_data, tmp_path) -> None:  # noqa: ANN001
    from tests.unit.fakes import FakeHMLoader

    loader = FakeHMLoader(tmp_path)
    raw = {
        "transactions": pd.read_csv(sample_hm_data["transactions"]),
        "articles": pd.read_csv(sample_hm_data["articles"]),
        "customers": pd.read_csv(sample_hm_data["customers"]),
    }
    data = loader.normalize(raw)

    # interactions: all H&M purchases normalized to 4-column schema
    assert set(data.interactions["event_type"]) == {"purchase"}
    assert data.interactions["user_id"].str.startswith("hm_user_").all()
    assert data.interactions["item_id"].str.startswith("hm_item_").all()
    assert pd.api.types.is_datetime64_any_dtype(data.interactions["timestamp"])
    assert list(data.interactions.columns) == ["user_id", "item_id", "event_type", "timestamp"]

    # items: internal 5-column schema, semantic text from H&M metadata
    assert list(data.items.columns) == ["item_id", "title", "category", "tags", "text_description"]
    jacket = data.items[data.items["item_id"] == "hm_item_10001"].iloc[0]
    assert jacket["title"] == "Black Slim Fit Jacket"
    assert jacket["category"] == "Jacket"
    assert "Solid" in jacket["tags"]
    # NaN detail_desc (no fabricated text) must not break flow
    missing = data.items[data.items["item_id"] == "hm_item_10002"].iloc[0]
    assert missing["text_description"]

    # optional metadata column is JSON and survives validation
    clean = validate_users(data.users)
    assert list(clean.columns) == ["user_id", "metadata"]
    row = clean[clean["user_id"] == "hm_user_12345"].iloc[0]
    meta = json.loads(row["metadata"])
    assert str(meta["postal_code"]) == "50200"
    assert meta["club_member_status"] == "ACTIVE"

    assert data.meta["source"] == "kaggle"
    assert data.meta["n_items"] == 3
