"""Two-Tower Retrieval model (Section 8.3-8.5).

User tower + Item tower → inner product similarity → top-K candidates.
Trains with contrastive loss (in-batch negative sampling).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TwoTowerModel(nn.Module):
    """Two-tower retrieval model.

    User tower: user id → embedding (or MLP on user features).
    Item tower: item id → embedding (or MLP on item features).
    Similarity = cosine of the two embeddings.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        n_user_features: int = 0,
        n_item_features: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim

        # User tower
        if n_user_features > 0:
            self.user_mlp = nn.Sequential(
                nn.Linear(n_user_features, emb_dim),
                nn.Dropout(dropout), nn.ReLU(), nn.Linear(emb_dim, emb_dim),
            )
        else:
            self.user_embedding = nn.Embedding(n_users, emb_dim)

        # Item tower
        if n_item_features > 0:
            self.item_mlp = nn.Sequential(
                nn.Linear(n_item_features, emb_dim),
                nn.Dropout(dropout), nn.ReLU(), nn.Linear(emb_dim, emb_dim),
            )
        else:
            self.item_embedding = nn.Embedding(n_items, emb_dim)

        self.init_weights()

    def init_weights(self) -> None:
        emb_range = 0.5 / self.emb_dim
        if hasattr(self, "user_embedding"):
            nn.init.uniform_(self.user_embedding.weight, -emb_range, emb_range)
        if hasattr(self, "item_embedding"):
            nn.init.uniform_(self.item_embedding.weight, -emb_range, emb_range)
        for name, module in self._modules.items():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def user_tower(
        self, user_idx: torch.Tensor, user_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        if hasattr(self, "user_mlp") and user_features is not None:
            h = self.user_mlp(user_features)
        else:
            h = self.user_embedding(user_idx)
        return F.normalize(h, p=2, dim=-1)

    def item_tower(
        self, item_idx: torch.Tensor, item_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        if hasattr(self, "item_mlp") and item_features is not None:
            h = self.item_mlp(item_features)
        else:
            h = self.item_embedding(item_idx)
        return F.normalize(h, p=2, dim=-1)

    def forward(
        self,
        user_idx: torch.Tensor,
        user_features: torch.Tensor | None = None,
        item_idx: torch.Tensor | None = None,
        item_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        u = self.user_tower(user_idx, user_features)
        i = self.item_tower(item_idx, item_features) if item_idx is not None else None
        return u, i


# ═══════════════════════════════════════════════════════════════
# Dataset for In-Batch Negative Sampling
# ═══════════════════════════════════════════════════════════════

class TwoTowerDataset(Dataset):
    """Dataset yielding (user_idx, pos_item_idx, neg_item_ids).

    Each __getitem__ returns one (user, positive_item, sampled_negative_items) tuple.
    The collate_fn reshapes these into a training batch.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        user_id_map: dict[str, int],
        item_id_map: dict[str, int],
        neg_samples: int = 1,
    ) -> None:
        self.interactions = interactions.reset_index(drop=True)
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.n_items = len(item_id_map)
        self.neg_samples = neg_samples
        # per-user positive items for negative sampling (avoid items user already saw)
        self.user_pos: dict[int, set] = {}
        for uid_str, group in interactions.groupby("user_id"):
            uid = user_id_map.get(str(uid_str), -1)
            if uid < 0:
                continue
            self.user_pos[uid] = {
                item_id_map[iid] for iid in group["item_id"] if iid in item_id_map
            }

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int) -> dict[str, int | list[int]]:
        row = self.interactions.iloc[idx]
        uid = self.user_id_map[row["user_id"]]
        pos_iid = self.item_id_map[row["item_id"]]

        seen = self.user_pos.get(uid, set())
        pool = list(set(range(self.n_items)) - seen)
        if not pool:
            pool = list(range(self.n_items))

        neg_ids = np.random.choice(pool, size=self.neg_samples, replace=False)
        return {
            "user_id": uid,
            "pos_item_id": pos_iid,
            "neg_item_ids": [int(x) for x in neg_ids],
        }


def two_tower_collate_fn(
    batch: list[dict[str, int | list[int]]],
) -> dict[str, torch.Tensor | list[int | list[int]]]:
    """Stack batch dictionaries into tensors."""
    return {
        "user_id": torch.tensor([b["user_id"] for b in batch], dtype=torch.long),
        "pos_item_id": torch.tensor([b["pos_item_id"] for b in batch], dtype=torch.long),
        "neg_item_ids": [b["neg_item_ids"] for b in batch],  # list-of-lists; reshaped in train step
    }


# ═══════════════════════════════════════════════════════════════
# Training Loop (contrastive loss, in-batch negatives)
# ═══════════════════════════════════════════════════════════════

def train_two_tower(
    model: TwoTowerModel,
    loader: DataLoader,
    epochs: int = 1,
    lr: float = 0.001,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Train with InfoNCE-style contrastive loss over positives + sampled negatives.

    For each user u with positive item p and negatives n_1..n_K, the loss is:

        -log softmax(score(u,p))  where the softmax is over
        {score(u,p), score(u,n_1), ... score(u,n_K)}

    This is equivalent (up to normalization) to BCE with one positive + K negatives,
    and matches the "in-batch negative sampling" description in spec §8.3.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    history: dict[str, list[float]] = {"train_loss": []}

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        for batch in loader:
            users = batch["user_id"].to(device)
            pos_items = batch["pos_item_id"].to(device)

            B = len(users)
            neg_lists: list[list[int]] = batch["neg_item_ids"]
            K = len(neg_lists[0]) if neg_lists else 0

            # --- Forward pass ---
            u_emb = model.user_tower(users)            # (B, dim)
            p_emb = model.item_tower(pos_items)         # (B, dim)
            s_pos = (u_emb * p_emb).sum(dim=-1)         # (B,)

            if K > 0:
                neg_items_flat = torch.tensor(
                    [n for sub in neg_lists for n in sub], dtype=torch.long, device=device
                )
                n_emb = model.item_tower(neg_items_flat)  # (B*K, dim)
                n_emb = n_emb.view(B, K, -1)               # (B, K, dim)
                # Score: u · n_k  → (B, K)
                s_neg = torch.bmm(u_emb.unsqueeze(1), n_emb.transpose(1, 2)).squeeze(1)
            else:
                s_neg = torch.empty(B, 0, device=device)

            # Concatenate positives and negatives, form logits matrix (B, 1+K)
            s_pos = s_pos.unsqueeze(-1)                    # (B, 1)
            logits = torch.cat([s_pos, s_neg], dim=1)      # (B, 1+K)
            targets = torch.zeros(B, dtype=torch.long, device=device)  # positives are column 0

            loss = nn.functional.cross_entropy(logits, targets)
            epoch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history["train_loss"].append(avg)
        print(f"  Epoch {epoch}/{epochs} | loss={avg:.4f}")

    return history


# ═══════════════════════════════════════════════════════════════
# Retrieval
# ═══════════════════════════════════════════════════════════════

def retrieve_top_k(
    model: TwoTowerModel,
    user_indices: list[int],
    n_items: int,
    k: int = 20,
    device: str = "cpu",
    exclude_seen: pd.DataFrame | None = None,
) -> dict[int, list[int]]:
    """Return top-k item indices per user using cosine similarity (in-memory).

    exclude_seen: optional DataFrame with user_id/item_id columns for this user
        set; items already interacted with by a user are filtered out.
    """
    model.eval()
    with torch.no_grad():
        # Precompute all item embeddings (normalized)
        item_indices = torch.arange(n_items, device=device)
        item_emb = model.item_tower(item_indices)              # (n_items, dim)
        item_emb = F.normalize(item_emb, p=2, dim=-1)

        # Precompute seen-item mask per user (using idx columns)
        seen_masks: dict[int, np.ndarray] = {}
        if exclude_seen is not None:
            for uid, group in exclude_seen.groupby("user_idx"):
                seen_masks[cast(int, uid)] = group["item_idx"].to_numpy()

        results: dict[int, list[int]] = {}
        for uid in user_indices:
            u_emb = model.user_tower(torch.tensor([uid], device=device))  # (1, dim)
            u_emb = F.normalize(u_emb, p=2, dim=-1)
            sims = item_emb @ u_emb.squeeze().unsqueeze(1)  # (n_items, 1)
            sims = sims.squeeze(1)                            # (n_items,)

            if uid in seen_masks:
                mask = seen_masks[uid].copy()
                sims[mask] = -1.0  # exclude seen items

            topk = sims.topk(k).indices.cpu().numpy()
            results[uid] = topk.tolist()
        return results


def retrieve_top_k_with_scores(
    model: TwoTowerModel,
    user_indices: list[int],
    n_items: int,
    k: int = 20,
    device: str = "cpu",
    exclude_seen: pd.DataFrame | None = None,
) -> dict[int, list[tuple[int, float]]]:
    """Like retrieve_top_k but also returns the cosine similarity per item.

    Returns {user_idx: [(item_idx, score), ...]} sorted by score descending.
    """
    model.eval()
    with torch.no_grad():
        item_indices = torch.arange(n_items, device=device)
        item_emb = model.item_tower(item_indices)              # (n_items, dim)
        item_emb = F.normalize(item_emb, p=2, dim=-1)

        seen_masks: dict[int, np.ndarray] = {}
        if exclude_seen is not None:
            for uid, group in exclude_seen.groupby("user_idx"):
                seen_masks[cast(int, uid)] = group["item_idx"].to_numpy()

        results: dict[int, list[tuple[int, float]]] = {}
        for uid in user_indices:
            u_emb = model.user_tower(torch.tensor([uid], device=device))  # (1, dim)
            u_emb = F.normalize(u_emb, p=2, dim=-1)
            sims = item_emb @ u_emb.squeeze().unsqueeze(1)  # (n_items, 1)
            sims = sims.squeeze(1)                            # (n_items,)

            if uid in seen_masks:
                mask = seen_masks[uid].copy()
                sims[mask] = -1.0  # exclude seen items

            topk = sims.topk(k)
            results[uid] = [
                (int(idx), float(score))
                for idx, score in zip(
                    topk.indices.cpu().numpy(),
                    topk.values.cpu().numpy(),
                    strict=False,
                )
            ]
        return results


# ═══════════════════════════════════════════════════════════════
# Evaluation (Recall/NDCG/HitRate @K)
# ═══════════════════════════════════════════════════════════════

def evaluate_retrieval(
    model: TwoTowerModel,
    eval_interactions: pd.DataFrame,
    user_id_map: dict[str, int],
    item_id_map: dict[str, int],
    train_interactions: pd.DataFrame,
    k_values: Sequence[int] = (10, 20),
    max_users: int = 500,
) -> dict[str, float]:
    """Compute Recall@K / NDCG@K / HitRate@K on held-out test interactions.

    Evaluation users = users in eval set that ALSO exist in training (we need a
    learned user embedding). Truth = items interacted in eval window that the user
    did NOT see in training.
    """
    from training.evaluate import hit_rate_at_k, ndcg_at_k, recall_at_k

    # Restrict to in-training users
    train_users = set(train_interactions["user_id"].map(lambda x: user_id_map.get(x, -1)))
    eval_with_idx = eval_interactions.copy()
    eval_with_idx["user_idx"] = eval_with_idx["user_id"].map(
        lambda x: user_id_map.get(x, -1)
    )
    eval_with_idx = eval_with_idx.loc[eval_with_idx["user_idx"].isin(list(train_users))]

    if len(eval_with_idx) == 0:
        zeros = {f"{m}@{k}": 0.0 for m in ("recall", "ndcg", "hit") for k in k_values}
        zeros["n_eval_users"] = 0.0
        return zeros

    # Build ground truth per user: eval items NOT seen during training
    train_user_items: dict[int, set] = {}
    for uid_str, group in train_interactions.groupby("user_id"):
        uid = user_id_map.get(str(uid_str), -1)
        if uid < 0:
            continue
        train_user_items[uid] = {
            item_id_map[iid] for iid in group["item_id"] if iid in item_id_map
        }

    # eval items mapped to indices, grouped by user
    eval_with_idx["item_idx"] = eval_with_idx["item_id"].map(
        lambda x: item_id_map.get(x, -1)
    )
    eval_with_idx = eval_with_idx.loc[eval_with_idx["item_idx"] >= 0]

    truth: dict[int, set] = {}
    for uid, group in eval_with_idx.groupby("user_idx"):
        uid_int = cast(int, uid)
        seen = train_user_items.get(uid_int, set())
        items = set(group["item_idx"]) - seen
        if items:
            truth[uid_int] = items

    # Sample eval users (deterministic: richest first)
    ranked_users = sorted(truth.keys(), key=lambda u: -len(truth[u]))
    eval_users = ranked_users[:max_users]

    n_items = len(item_id_map)
    device_str = next(model.parameters()).device.type

    # Build exclude_seen DataFrame using integer indices (user_idx, item_idx)
    # so retrieve_top_k can mask out seen items during retrieval.
    exclude_seen: pd.DataFrame = train_interactions.copy()
    exclude_seen["user_idx"] = exclude_seen["user_id"].map(lambda x: user_id_map.get(x, -1))
    exclude_seen["item_idx"] = exclude_seen["item_id"].map(lambda x: item_id_map.get(x, -1))
    exclude_seen = exclude_seen.loc[exclude_seen["item_idx"] >= 0, ["user_idx", "item_idx"]]

    recs = retrieve_top_k(
        model, eval_users, n_items, k=max(k_values),
        device=device_str,
        exclude_seen=exclude_seen,
    )

    metrics: dict[str, float] = {}
    for k in k_values:
        rec_vals, ndcg_vals, hit_vals = [], [], []
        for uid in eval_users:
            ground = truth.get(uid, set())
            if not ground:
                continue
            recommended = recs.get(uid, [])[:k]
            rec_vals.append(recall_at_k([str(x) for x in recommended], {str(x) for x in ground}, k))
            ndcg_vals.append(ndcg_at_k([str(x) for x in recommended], {str(x) for x in ground}, k))
            hit_vals.append(
                hit_rate_at_k([str(x) for x in recommended], {str(x) for x in ground}, k)
            )
        metrics[f"recall@{k}"] = float(np.mean(rec_vals)) if rec_vals else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_vals)) if ndcg_vals else 0.0
        metrics[f"hit@{k}"] = float(np.mean(hit_vals)) if hit_vals else 0.0
    metrics["n_eval_users"] = float(len(eval_users))
    return metrics
