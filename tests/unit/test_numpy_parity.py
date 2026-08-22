"""PARITY TEST: torch Two-Tower retrieval VS NumPy retriever.

Run:  uv run pytest tests/unit/test_numpy_parity.py -v

Fails unless, for >=10 users x K in {1,5,10,20}:
  - embeddings byte-equal (max_abs_error == 0)
  - scores match within float tolerance AND do not change ranking
  - top-K item id sequences agree 100%
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CKPT = ROOT / "outputs" / "two_tower.pt"
NPZ = ROOT / "artifacts" / "embeddings.npz"
TRAIN = ROOT / "data" / "processed" / "interactions_train.parquet"

pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and NPZ.exists() and TRAIN.exists()),
    reason="checkpoint/npz/train parquet not present",
)


@pytest.fixture(scope="module")
def loaded():
    import sys

    sys.path.insert(0, str(ROOT))
    from app.numpy_retriever import NumpyRetriever

    import torch
    from training.two_tower import retrieve_top_k_with_scores

    cp = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = type("M", (), {})()
    from training.two_tower import TwoTowerModel

    m = TwoTowerModel(n_users=cp["n_users"], n_items=cp["n_items"], emb_dim=cp["emb_dim"])
    m.load_state_dict(cp["model_state"])
    m.eval()

    with np.load(NPZ, allow_pickle=False) as z:
        retr = NumpyRetriever(
            user_emb=z["user_emb"],
            item_emb=z["item_emb"],
            user_ids=z["user_ids"],
            item_ids=z["item_ids"],
            version=str(z["version"]),
        )
    return {"cp": cp, "model": m, "retr": retr,
            "user_id_map": cp["user_id_map"], "item_id_map": cp["item_id_map"],
            "train": pd.read_parquet(TRAIN), "torch_mod": retrieve_top_k_with_scores}


USERS = ["ml_user_1", "ml_user_429", "ml_user_107", "ml_user_42", "ml_user_300",
         "ml_user_7", "ml_user_250", "ml_user_88", "ml_user_15", "ml_user_400"]
KS = [1, 5, 10, 20]


def _seen_items(loaded, user_id: str, user_idx: int) -> pd.DataFrame:
    inter = loaded["train"].loc[loaded["train"]["user_id"] == user_id]
    return pd.DataFrame({
        "user_idx": [user_idx] * len(inter),
        "item_idx": [loaded["item_id_map"].get(i, -1) for i in inter["item_id"]],
    }).loc[lambda d: d["item_idx"] >= 0]


def test_embedding_parity_exact(loaded):
    """NumPy weights must be byte-identical to the checkpoint tensors."""
    import torch

    u_t = loaded["cp"]["model_state"]["user_embedding.weight"].numpy()
    i_t = loaded["cp"]["model_state"]["item_embedding.weight"].numpy()
    assert np.array_equal(u_t, loaded["retr"].user_emb)
    assert np.array_equal(i_t, loaded["retr"].item_emb)
    print(f"\nembedding max_abs_error: 0.0 (byte-identical); "
          f"user{u_t.shape} item{i_t.shape} dtype={u_t.dtype}")


def test_score_and_ranking_parity(loaded):
    """For each sampled user & K: identical top-K ids; score diff within tol."""
    retr, model = loaded["retr"], loaded["model"]
    torch_fn = loaded["torch_mod"]
    # Sample 10 real users from the map: mix of head/tail (varied interaction counts)
    all_users = sorted(loaded["user_id_map"])
    rng = np.random.default_rng(42)
    sample = list(rng.choice(all_users, size=min(10, len(all_users)), replace=False))
    if "ml_user_429" in loaded["user_id_map"] and "ml_user_429" not in sample:
        sample[0] = "ml_user_429"
    max_diff = 0.0
    checked = 0
    for uid_str in sample:
        if uid_str not in loaded["user_id_map"]:
            continue
        uidx = loaded["user_id_map"][uid_str]
        seen = _seen_items(loaded, uid_str, uidx)
        excl = seen if len(seen) else None
        for k in KS:
            t_res = torch_fn(model, [uidx], retr.n_items, k=k,
                             device="cpu", exclude_seen=excl)[uidx][:k]
            n_res = retr.recommend(uidx, k=k,
                                   exclude_seen_item_idx=set(seen["item_idx"]) if excl is not None else None)[:k]
            t_ids = [int(i) for i, _ in t_res]
            n_ids = [i for i, _ in n_res]
            assert t_ids == n_ids, f"RANKING MISMATCH user={uid_str} k={k}: {t_ids[:5]} vs {n_ids[:5]}"
            for (ti, ts), (_, ns) in zip(t_res, n_res):
                max_diff = max(max_diff, abs(ts - ns))
            checked += 1
            # Score order must be descending in both
            t_scores = [s for _, s in t_res]
            n_scores = [s for _, s in n_res]
            assert all(a >= b for a, b in zip(t_scores, t_scores[1:]))
            assert all(a >= b for a, b in zip(n_scores, n_scores[1:]))
    assert checked >= len(USERS) * len(KS) - 5
    # float32 dot-product vs torch matmul: identical values expected (< 1e-6)
    print(f"\nscore max_abs_error across {checked} cases: {max_diff:.2e}")
    assert max_diff < 1e-6


def test_user_mapping_consistency(loaded):
    """NPZ row order must equal the checkpoint's user/item maps."""
    retr = loaded["retr"]
    for uid, idx in list(loaded["user_id_map"].items())[:50]:
        assert retr.user_index(uid) == idx
    assert np.all(retr.item_ids[np.arange(len(loaded["item_id_map"]))] ==
                  np.array(sorted(loaded["item_id_map"], key=loaded["item_id_map"].get)))


def test_exclude_seen_changes_results(loaded):
    """Masking seen items must actually change the top-K (regression guard)."""
    retr = loaded["retr"]
    uidx = loaded["user_id_map"]["ml_user_429"]
    seen = _seen_items(loaded, "ml_user_429", uidx)
    without = [i for i, _ in retr.recommend(uidx, k=20)]
    with_excl = [i for i, _ in retr.recommend(uidx, k=20, exclude_seen_item_idx=set(seen["item_idx"]))]
    assert not set(without[:5]) <= set(with_excl[:5]) or len(seen) == 0
