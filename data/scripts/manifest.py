"""Dataset manifest (Section 5.6) — versioning metadata written at ingest time."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ManifestDataset(BaseModel):
    name: str
    source: str
    version: str
    row_count: int = 0
    downloaded_at: datetime | None = None
    checksum: str | None = None


class DatasetManifest(BaseModel):
    dataset: ManifestDataset
    schema_fields: dict = Field(default_factory=dict)
    splits: dict[str, int] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    pipeline_version: str = "ingest-v1"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        return cls.model_validate(yaml.safe_load(path.read_text()))


def sha256_of(path: Path) -> str | None:
    """Checksum of a single file; returns None for directories (too heavy for big trees)."""
    if path.is_file():
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    return None
