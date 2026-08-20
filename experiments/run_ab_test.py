"""Run an offline A/B test comparing Two-Tower vs. Popularity baseline.

Usage:
    python3 experiments/run_ab_test.py [--epochs N] [--users N]

Produces results in experiments/results/ab_test_<timestamp>.json
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from experiments.ab_testing import Experiment
from training.evaluate import build_ground_truth, select_eval_users
from training.two_tower import (
    TwoTowerDataset,
    TwoTowerModel,
    retrieve_top_k,
    train_two_tower,
    two_tower_collate_fn,
)

API_HOST = "http://localhost:8013"


def make_recommenders(train_df, test_df, k=20, max_users=2000, epochs=3):
    """Build recommender functions for each variant in the A/B test."""

    # --- Variant 1: Two-Tower ---
    random.seed(42)
    test_users = test_df["user_id"].unique()
    sampled = random.sample(list(test_users), min(max_users, len(test_users)))
    train_sub = train_df[train_df["user_id"].isin(sampled)]
    test_sub = test_df[test_df["user_id"].isin(sampled)]
    common = set(train_sub["user_id"]) & set(test_sub["user_id"])
    train_sub = train_df[train_df["user_id"].isin(common)]
    test_sub = test_df[test_df["user_id"].isin(common)]

    items = sorted(set(train_sub["item_id"]) | set(test_sub["item_id"]))
    users = sorted(common)
    item_id_map = {iid: i for i, iid in enumerate(items)}
    user_id_map = {uid: i for i, uid in enumerate(users)}

    dataset = TwoTowerDataset(train_sub, user_id_map, item_id_map, neg_samples=4)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=two_tower_collate_fn)

    model = TwoTowerModel(len(user_id_map), len(item_id_map), emb_dim=64)
    train_two_tower(model, loader, epochs=epochs, lr=0.001, device="cpu")

    inv_item = {i: iid for iid, i in item_id_map.items()}

    def two_tower_recommender(user_id: str, topk: int = 20) -> list[str]:
        uid = user_id_map.get(user_id, -1)
        if uid < 0:
            return []
        rec_indices = retrieve_top_k(model, [uid], len(item_id_map), k=topk, device="cpu")[uid]
        rec_indices = rec_indices[:topk]
        return [inv_item[idx] for idx in rec_indices if idx in inv_item]

    # --- Variant 2: Popularity baseline ---
    from training.common import build_weighted_counts
    weights = build_weighted_counts(train_sub)
    weighted = train_sub.assign(w=weights.values)
    pop_scores = weighted.groupby("item_id")["w"].sum().sort_values(ascending=False)
    pop_items = pop_scores.index.tolist()

    def popularity_recommender(user_id: str, topk: int = 20) -> list[str]:
        return pop_items[:topk]

    return two_tower_recommender, popularity_recommender, item_id_map, user_id_map, train_sub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=5000)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    print("Loading data...")
    train_df = pd.read_parquet("data/processed/interactions_train.parquet")
    test_df = pd.read_parquet("data/processed/interactions_test.parquet")

    print("Building recommenders (training Two-Tower)...")
    t0 = time.time()
    tt_rec, pop_rec, item_map, user_map, train_sub = make_recommenders(
        train_df, test_df, k=args.k, epochs=3
    )
    print(f"  Built in {time.time() - t0:.1f}s")

    # Create experiment
    exp = Experiment("rec-ab-test")
    exp.add_variant("two-tower", tt_rec, weight=0.5)
    exp.add_variant("popularity", pop_rec, weight=0.5)
    exp.save()
    print(f"Experiment: {exp.name}")
    print(f"  Variants: {[v.name for v in exp.variants]}")

    # Evaluate each variant on held-out users using ground truth
    catalog = set(item_map.keys()) if item_map else None
    truth = build_ground_truth(test_df, train_df, catalog=catalog)
    eval_users = select_eval_users(truth, max_users=args.users, restrict=set(user_map.keys()))
    print(f"Evaluating on {len(eval_users)} users")

    k = args.k
    results = {}
    for v in exp.variants:
        r_vals, n_vals, h_vals = [], [], []
        from training.evaluate import hit_rate_at_k, ndcg_at_k, recall_at_k
        for uid in eval_users:
            gt = truth.get(uid, set())
            if not gt:
                continue
            recs = v.recommender(uid, k)
            rec_strs = [str(x) for x in recs[:k]]
            gt_strs = {str(x) for x in gt}
            r_vals.append(recall_at_k(rec_strs, gt_strs, k))
            n_vals.append(ndcg_at_k(rec_strs, gt_strs, k))
            h_vals.append(hit_rate_at_k(rec_strs, gt_strs, k))
        results[v.name] = {
            f"recall@{k}": float(np.mean(r_vals)) if r_vals else 0.0,
            f"ndcg@{k}": float(np.mean(n_vals)) if n_vals else 0.0,
            f"hit@{k}": float(np.mean(h_vals)) if h_vals else 0.0,
            "n_users": len(r_vals),
            "per_user_recall": r_vals,
            "per_user_hit": h_vals,
        }
        print(
            f"  {v.name}: recall@{k}={results[v.name][f'recall@{k}']:.4f}"
            f"  hit@{k}={results[v.name][f'hit@{k}']:.4f}"
        )

    # Statistical significance (Welch's t-test on per-user recall)
    from scipy import stats as sp_stats
    r1 = np.array(results["two-tower"]["per_user_recall"])
    r2 = np.array(results["popularity"]["per_user_recall"])
    if len(r1) > 1 and len(r2) > 1:
        t_stat, p_val = sp_stats.ttest_ind(r1, r2, equal_var=False)
        print(f"\n  Welch t-test (recall@{k}): t={t_stat:.3f}, p={p_val:.4f}")

    # Clean up per-user arrays before saving
    for vname in results:
        results[vname].pop("per_user_recall", None)
        results[vname].pop("per_user_hit", None)

    print("\nDone!")


if __name__ == "__main__":
    main()
