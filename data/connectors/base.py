"""Dataset loader interface and registry (Section 5.3, 5.4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class LoadedData:
    interactions: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame
    meta: dict = field(default_factory=dict)
    source_path: Path | None = None


class BaseDatasetLoader(ABC):
    """A loader normalizes a source dataset into the internal schema."""

    name: str = ""
    version: str = ""

    def __init__(self, version: str | None = None) -> None:
        if version is not None:
            self.version = version

    @abstractmethod
    def download(self, raw_dir: Path) -> Path:
        """Fetch the source data into raw_dir and return the entry point path."""
        raise NotImplementedError

    @abstractmethod
    def load_raw(self, source: Path) -> dict[str, pd.DataFrame]:
        """Load source files into raw dataframes keyed by name."""
        raise NotImplementedError

    @abstractmethod
    def normalize(self, raw: dict[str, pd.DataFrame]) -> LoadedData:
        """Convert raw tables into the internal schema (Section 5.4)."""
        raise NotImplementedError

    def load(self, raw_dir: Path) -> LoadedData:
        source = self.download(raw_dir)
        raw = self.load_raw(source)
        data = self.normalize(raw)
        data.source_path = source
        return data


_REGISTRY: dict[str, type[BaseDatasetLoader]] = {}


def register(name: str):
    """Decorator to register a loader implementation."""

    def decorator(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_loader(
    name: str, version: str | None = None, max_rows: int | None = None
) -> BaseDatasetLoader:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown dataset {name!r}; available: {available}")
    return _REGISTRY[name](version=version, max_rows=max_rows)


def available_loaders() -> list[str]:
    return sorted(_REGISTRY)
