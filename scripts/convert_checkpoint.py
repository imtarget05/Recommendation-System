"""Convert two_tower.pt checkpoint → artifacts/embeddings.npz (NumPy-only serving).

Run ONCE offline (needs torch):
    uv run python -m scripts.convert_checkpoint

Output NPZ keys:
    user_emb      (n_users, dim) float32  — raw (un-normalized) user embeddings
    item_emb      (n_items, dim) float32  — raw (un-normalized) item embeddings
    user_ids      (n_users,)     <U…      — row order of user_emb
    item_ids      (n_items,)     <U…      — row order of item_emb
    version       ()             <U…      — model version string

Normalization is applied at query time by the NumPy retriever so the stored
vectors stay byte-equivalent to the torch checkpoint weights.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def convert(checkpoint_path: str = "outputs/two_tower.pt", out_dir: str = "artifacts") -> Path:
    import torch

    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = cp["model_state"]

    user_emb = state["user_embedding.weight"].cpu().numpy().astype(np.float32)
    item_emb = state["item_embedding.weight"].cpu().numpy().astype(np.float32)

    user_id_map: dict[str, int] = cp["user_id_map"]
    item_id_map: dict[str, int] = cp["item_id_map"]

    # Sanity: map indices must cover the embedding rows exactly.
    assert len(user_id_map) == user_emb.shape[0], "user map/emb mismatch"
    assert len(item_id_map) == item_emb.shape[0], "item map/emb mismatch"

    user_ids = np.array(sorted(user_id_map, key=lambda k: user_id_map[k]), dtype=np.str_)
    item_ids = np.array(sorted(item_id_map, key=lambda k: item_id_map[k]), dtype=np.str_)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "embeddings.npz"
    np.savez_compressed(
        dest,
        user_emb=user_emb,
        item_emb=item_emb,
        user_ids=user_ids,
        item_ids=item_ids,
        version=str(cp.get("version", "two-tower-v1")),
    )

    print(f"user_emb: {user_emb.shape} {user_emb.dtype}")
    print(f"item_emb: {item_emb.shape} {item_emb.dtype}")
    print(f"version : {cp.get('version')}")
    print(f"saved   : {dest} ({dest.stat().st_size / 1e6:.2f} MB)")
    return dest


if __name__ == "__main__":
    convert(
        os.environ.get("MODEL_PATH", "outputs/two_tower.pt"),
        os.environ.get("ARTIFACTS_DIR", "artifacts"),
    )
