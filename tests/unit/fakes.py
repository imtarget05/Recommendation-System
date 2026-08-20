from __future__ import annotations

from pathlib import Path

from data.connectors.kaggle_loader import HMLoader
from data.connectors.movielens_loader import MovieLensLoader


class FakeMovielensLoader(MovieLensLoader):
    """MovieLens loader with download/network disabled for tests."""

    def __init__(self, source_dir: Path, version: str = "ml-latest-small") -> None:
        super().__init__(version=version)
        self._source_dir = source_dir

    def download(self, raw_dir: Path) -> Path:
        return self._source_dir / f"{self.version}.zip"


class FakeHMLoader(HMLoader):
    """H&M loader with Kaggle download/credentials disabled for tests.

    `download()` points at an on-disk directory laid out like the Kaggle cache.
    """

    def __init__(self, source_dir: Path, version: str = "v1") -> None:
        super().__init__(version=version)
        self._source_dir = source_dir

    def download(self, raw_dir: Path) -> Path:
        return self._source_dir
