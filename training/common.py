"""Shared training utilities: data loading, implicit-feedback matrix, artifact IO."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from app.config import load_config

MODELS_ROOT = Path("models")


def load_processed(split_names: tuple[str, ...] = ("train", "validation", "test")) -> dict:
    """Load processed parquet splits + items + users (from Phase 1)."""
    cfg = load_config()
    processed = cfg.data.processed_dir
    missing = [s for s in split_names if not (processed / f"interactions_{s}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"missing processed splits {missing}; run the ingest CLI first"
        )
    out = {s: pd.read_parquet(processed / f"interactions_{s}.parquet") for s in split_names}
    out["items"] = pd.read_parquet(processed / "items.parquet")
    out["users"] = pd.read_parquet(processed / "users.parquet")
    return out


def dataset_version() -> str:
    cfg = load_config()
    manifest = cfg.data.manifests_dir / "dataset_manifest.yaml"
    if manifest.exists():
        raw = yaml.safe_load(manifest.read_text())
        d = raw.get("dataset", {})
        return f"{d.get('name')}-{d.get('version')}"
    return "unknown"


def git_commit() -> str:
    try:
        return subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Typed column accessor (pandas stubs type ``df[name]`` as ``Series | DataFrame``)."""
    return cast(pd.Series, df[name])


def build_implicit_matrix(
    interactions: pd.DataFrame,
    weights: pd.Series | None = None,
    shape: tuple[int, int] | None = None,
) -> tuple[sparse.csr_matrix, dict, dict]:
    """Build a users x items weighted implicit matrix from interactions.

    Returns (matrix, user_index, item_index) where indexes map id -> row/col.
    """
    users = pd.unique(_col(interactions, "user_id"))
    items = pd.unique(_col(interactions, "item_id"))
    user_index = {u: i for i, u in enumerate(users)}
    item_index = {i: j for j, i in enumerate(items)}
    if shape is None:
        shape = (len(users), len(items))
    w = weights if weights is not None else np.ones(len(interactions))
    matrix = sparse.csr_matrix(
        (
            np.asarray(w, dtype=np.float64),
            (
                _col(interactions, "user_id").map(user_index).to_numpy(),
                _col(interactions, "item_id").map(item_index).to_numpy(),
            ),
        ),
        shape=shape,
    )
    return matrix, user_index, item_index


def build_weighted_counts(interactions: pd.DataFrame, config: Any = None) -> pd.Series:
    """event-weight per interaction (Section 6.2)."""
    cfg = config or load_config()
    return _col(interactions, "event_type").map(
        lambda e: getattr(cfg.interaction_weights, e, 1.0)
    ).astype(float)


class ModelArtifact:
    """Saves a model directory + metadata.yaml under models/ (Section 33)."""

    def __init__(
        self,
        name: str,
        version: str,
        metrics: dict,
        hyperparameters: dict,
        extra: dict | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.metrics = metrics
        self.hyperparameters = hyperparameters
        self.extra = extra or {}
        self.dir = MODELS_ROOT / name / version
        self.metadata = {
            "model_name": name,
            "model_version": version,
            "dataset_version": dataset_version(),
            "code_commit": git_commit(),
            "training_timestamp": now_utc(),
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "status": "candidate",
            "notes": extra,
        }

    def path(self) -> Path:
        return self.dir

    def save(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.dir / "metadata.yaml"
        meta_path.write_text(yaml.safe_dump(self.metadata, sort_keys=False))
        return self.dir

    def write_artifact(self, filename: str, data) -> Path:  # noqa: ANN001
        target = self.dir / filename
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data.to_parquet(target.with_suffix(".parquet"))
        elif isinstance(data, dict):
            target.write_text(json.dumps(data, indent=2, default=str))
        else:
            raise TypeError(f"unsupported artifact type {type(data)}")
        return target


def add_to_registry(
    artifact: ModelArtifact, registry_path: Path = MODELS_ROOT / "registry.json"
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(registry_path.read_text()) if registry_path.exists() else []
    entries.append(artifact.metadata)
    registry_path.write_text(json.dumps(entries, indent=2, default=str))


def set_registry_status(model_name: str, version: str, status: str) -> None:
    """Promote/demote a registered model (Section 33 lifecycle)."""
    registry_path = MODELS_ROOT / "registry.json"
    entries = json.loads(registry_path.read_text())
    for e in entries:
        if e["model_name"] == model_name and e["model_version"] == version:
            e["status"] = status
    registry_path.write_text(json.dumps(entries, indent=2, default=str))


def checksum_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]
