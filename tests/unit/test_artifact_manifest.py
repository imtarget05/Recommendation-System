"""P2 regression tests: versioned artifact manifest verification (cases A-G)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import api.main as api_mod


def _make_world(tmp_path: Path, n_users: int = 5, n_items: int = 7):
    """Build a consistent npz + parquet + manifest triple in tmp_path."""
    rng = np.random.default_rng(0)
    user_ids = [f"ml_user_{i}" for i in range(n_users)]
    item_ids = [f"ml_{i}" for i in range(n_items)]
    user_emb = rng.normal(size=(n_users, 8)).astype(np.float32)
    item_emb = rng.normal(size=(n_items, 8)).astype(np.float32)

    emb_path = tmp_path / "embeddings.npz"
    np.savez(emb_path, user_emb=user_emb, item_emb=item_emb,
             user_ids=np.array(user_ids), item_ids=np.array(item_ids))

    items_df = pd.DataFrame({"item_id": item_ids, "title": [f"T{i}" for i in range(n_items)]})
    train_df = pd.DataFrame({
        "user_id": [user_ids[i % n_users] for i in range(10)],
        "item_id": [item_ids[i % n_items] for i in range(10)],
        "event_type": ["view"] * 10,
        "rating": [4.0] * 10,
    })
    items_df.to_parquet(tmp_path / "items.parquet")
    train_df.to_parquet(tmp_path / "interactions_train.parquet")

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = {
        "schema_version": 1,
        "release_version": "recsys-test",
        "user_count": n_users,
        "item_count": n_items,
        "user_ids_sha256": hashlib.sha256("\n".join(sorted(user_ids)).encode()).hexdigest(),
        "item_ids_sha256": hashlib.sha256("\n".join(sorted(item_ids)).encode()).hexdigest(),
        "artifacts": {
            "embeddings.npz": {"sha256": sha(emb_path)},
            "items.parquet": {"sha256": sha(tmp_path / "items.parquet")},
            "interactions_train.parquet": {"sha256": sha(tmp_path / "interactions_train.parquet")},
        },
    }
    return tmp_path, emb_path, manifest


def _setup(monkeypatch, tmp_path: Path, emb_path: Path):
    monkeypatch.setattr(api_mod, "MODEL_PATH", str(emb_path))
    monkeypatch.setattr(api_mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api_mod, "MANIFEST_URL", "")
    state = api_mod.AppState()
    items_df = pd.read_parquet(tmp_path / "items.parquet")
    train_df = pd.read_parquet(tmp_path / "interactions_train.parquet")
    state.items_df = items_df
    state.train_df = train_df
    state.item_id_map = {iid: i for i, iid in enumerate(items_df["item_id"])}
    state.idx_to_item = {v: k for k, v in state.item_id_map.items()}
    state.n_items = len(state.item_id_map)
    state.user_id_map = {uid: i for i, uid in enumerate(sorted(train_df["user_id"].unique()))}
    monkeypatch.setattr(api_mod, "state", state)
    return api_mod.load_retriever(str(emb_path))


def test_manifest_valid_passes(monkeypatch, tmp_path):
    """Case A: consistent manifest + artifacts → PASS."""
    tmp, emb, manifest = _make_world(tmp_path)
    (tmp / "manifest.json").write_text(json.dumps(manifest))
    api_mod._validate_artifacts(_setup(monkeypatch, tmp, emb))  # must not raise
