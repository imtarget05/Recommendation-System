"""NumPy-only Two-Tower retriever (production serving path — zero torch).

Replicates training/two_tower.py::retrieve_top_k_with_scores exactly:
    u      = user_emb[uid] / ||user_emb[uid]||
    I      = item_emb / ||item_emb||        (row-wise)
    scores = I @ u                          (cosine similarity)
    scores[seen] = -1.0                     (exclude seen items)
    top-k   sorted by score descending

Embeddings are stored raw (un-normalized), identical to the torch checkpoint
weights; normalization happens at query time so stored bytes match the .pt file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NumpyRetriever:
    """Pure-NumPy two-tower retrieval over converted checkpoint embeddings."""

    user_emb: np.ndarray  # (n_users, dim) float32, raw
    item_emb: np.ndarray  # (n_items, dim) float32, raw
    user_ids: np.ndarray  # (n_users,) str, row order of user_emb
    item_ids: np.ndarray  # (n_items,) str, row order of item_emb
    version: str = "two-tower-v1"

    def __post_init__(self) -> None:
        self.user_emb = np.ascontiguousarray(self.user_emb, dtype=np.float32)
        self.item_emb = np.ascontiguousarray(self.item_emb, dtype=np.float32)
        # Pre-normalize item rows once (equivalent to torch F.normalize per query,
        # since item table is static). Norm-zero rows stay all-zero like torch's
        # F.normalize (eps clamp).
        norms = np.linalg.norm(self.item_emb, axis=1, keepdims=True)
        self._item_unit = self.item_emb / np.maximum(norms, 1e-12)

    @property
    def n_users(self) -> int:
        return int(self.user_emb.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.item_emb.shape[0])

    def user_index(self, user_id: str) -> int:
        """Row index for a user id (same mapping the torch path uses)."""
        idx = int(np.searchsorted(self.user_ids, user_id))
        if idx >= len(self.user_ids) or self.user_ids[idx] != user_id:
            raise KeyError(user_id)
        return idx

    def has_user(self, user_id: str) -> bool:
        try:
            self.user_index(user_id)
            return True
        except KeyError:
            return False

    def recommend(
        self,
        user_idx: int,
        k: int = 20,
        exclude_seen_item_idx: list[int] | set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Top-k [(item_idx, score), ...] sorted by score desc.

        Mirrors retrieve_top_k_with_scores: cosine scores, seen items masked to
        -1.0 before top-k, deterministic descending order (numpy argsort is a
        stable sort here so equal scores keep index order, matching torch's
        behavior on distinct-valued tables used in parity tests).
        """
        u = self.user_emb[user_idx]
        u_norm = float(np.linalg.norm(u))
        if u_norm > 0:
            u = u / max(u_norm, 1e-12)
        scores = self._item_unit @ u  # (n_items,) float32 cosine

        if exclude_seen_item_idx:
            scores[np.fromiter(exclude_seen_item_idx, dtype=np.int64)] = np.float32(-1.0)

        k = min(k, self.n_items)
        # Descending top-k. argpartition then stable sort the slice by -score.
        if k < self.n_items:
            part = np.argpartition(scores, -k)[-k:]
        else:
            part = np.arange(self.n_items)
        order = part[np.argsort(-scores[part], kind="stable")]
        return [(int(i), float(scores[i])) for i in order]


def load_retriever(npz_path: str = "artifacts/embeddings.npz") -> NumpyRetriever:
    """Load a retriever from an NPZ produced by scripts/convert_checkpoint.py."""
    with np.load(npz_path, allow_pickle=False) as z:
        return NumpyRetriever(
            user_emb=z["user_emb"],
            item_emb=z["item_emb"],
            user_ids=z["user_ids"],
            item_ids=z["item_ids"],
            version=str(z["version"]),
        )


def exclude_seen_for_user(
    train_df: pd.DataFrame,
    user_id: str,
    user_idx: int,
    item_index_of: dict[str, int],
) -> list[int]:
    """Item indices this user interacted with in training (mirrors the API's
    exclude_seen construction: unknown items dropped via item_idx >= 0 filter)."""
    inter = train_df.loc[train_df["user_id"] == user_id]
    return [
        idx
        for iid in inter["item_id"]
        if (idx := item_index_of.get(str(iid), -1)) >= 0
    ]
