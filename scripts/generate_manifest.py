"""Generate a versioned artifact manifest (P2 — artifact integrity).

Reads the production artifacts, validates their internal consistency, computes
SHA256 hashes, and writes artifacts/manifest.json.

Version scheme (deterministic, not timestamp-based):
    recsys-<git_short_sha>
The git SHA pins the *code* that produced the artifacts; the per-file SHA256
pins the *content*. A release is reproducible when both match. Timestamps are
deliberately avoided as the primary key because they are not reproducible.

Fails (exit 1) if the artifacts are already inconsistent — never "fixes" data.

Usage:
    uv run python scripts/generate_manifest.py \
        [--embeddings artifacts/embeddings.npz] \
        [--data-dir data/processed] \
        [--out artifacts/manifest.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="artifacts/embeddings.npz")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="artifacts/manifest.json")
    args = ap.parse_args()

    emb_path = Path(args.embeddings)
    items_path = Path(args.data_dir) / "items.parquet"
    train_path = Path(args.data_dir) / "interactions_train.parquet"
    for p in (emb_path, items_path, train_path):
        if not p.exists():
            print(f"ERROR: missing artifact {p}", file=sys.stderr)
            return 1

    npz = np.load(emb_path)
    user_emb: np.ndarray = npz["user_emb"]
    item_emb: np.ndarray = npz["item_emb"]
    user_ids: np.ndarray = npz["user_ids"]
    item_ids: np.ndarray = npz["item_ids"]

    items_df = pd.read_parquet(items_path)
    train_df = pd.read_parquet(train_path)
    ds_user_ids = set(train_df["user_id"].astype(str))
    ds_item_ids = set(items_df["item_id"].astype(str))

    # ── Pre-write validation: refuse to manifest inconsistent artifacts ──
    errors: list[str] = []
    if len(user_ids) != user_emb.shape[0]:
        errors.append(f"user_ids({len(user_ids)}) != user_emb rows({user_emb.shape[0]})")
    if len(item_ids) != item_emb.shape[0]:
        errors.append(f"item_ids({len(item_ids)}) != item_emb rows({item_emb.shape[0]})")
    if set(map(str, user_ids)) != ds_user_ids:
        errors.append(
            f"user id set mismatch vs dataset (npz={len(set(user_ids))} ds={len(ds_user_ids)})"
        )
    if set(map(str, item_ids)) != ds_item_ids:
        errors.append(
            f"item id set mismatch vs dataset (npz={len(set(item_ids))} ds={len(ds_item_ids)})"
        )
    if user_emb.shape[1] != item_emb.shape[1]:
        errors.append(f"emb dim mismatch user={user_emb.shape[1]} item={item_emb.shape[1]}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_version": f"recsys-{git_short_sha()}",
        "model_version": "two-tower-numpy-v1",
        "dataset_version": "movielens-small-v1",
        "embedding_version": "two-tower-64d",
        "semantic_model_version": "sentence-transformers/all-MiniLM-L6-v2 (onnx)",
        "user_count": int(user_emb.shape[0]),
        "item_count": int(item_emb.shape[0]),
        "recommendation_embedding_dim": int(user_emb.shape[1]),
        "semantic_embedding_dim": 384,
        "user_ids_sha256": hashlib.sha256(
            "\n".join(sorted(map(str, user_ids))).encode()
        ).hexdigest(),
        "item_ids_sha256": hashlib.sha256(
            "\n".join(sorted(map(str, item_ids))).encode()
        ).hexdigest(),
        "artifacts": {
            "embeddings.npz": {
                "sha256": sha256_file(emb_path),
                "size_bytes": emb_path.stat().st_size,
            },
            "items.parquet": {
                "sha256": sha256_file(items_path),
                "size_bytes": items_path.stat().st_size,
            },
            "interactions_train.parquet": {
                "sha256": sha256_file(train_path),
                "size_bytes": train_path.stat().st_size,
            },
        },
    }

    out = Path(args.out)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest written: {out} (release={manifest['release_version']}, "
          f"users={manifest['user_count']}, items={manifest['item_count']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
