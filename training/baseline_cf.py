"""Collaborative-filtering baseline (Section 8.2): implicit ALS via `implicit`.

Usage:
    uv run python -m training.baseline_cf [--eval-users N] [--factors 64] [--iterations 15]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from implicit.als import AlternatingLeastSquares

from data.preprocessing.normalize import validate_interactions
from training.common import (
    ModelArtifact,
    add_to_registry,
    build_implicit_matrix,
    build_weighted_counts,
    dataset_version,
    load_processed,
)
from training.evaluate import (
    build_ground_truth,
    coverage,
    diversity_category,
    evaluate_ranking_metrics,
    select_eval_users,
    summarize,
)
from training.results import record_experiment

CONFIG = Path("training/configs/baselines.yaml")


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    cf_cfg = cfg["cf"]
    eval_cfg = cfg["eval"]

    parser = argparse.ArgumentParser(description="Train + evaluate the CF (ALS) baseline")
    parser.add_argument("--eval-users", type=int, default=eval_cfg["max_users"])
    parser.add_argument("--factors", type=int, default=cf_cfg["factors"])
    parser.add_argument("--iterations", type=int, default=cf_cfg["iterations"])
    parser.add_argument("--alpha", type=float, default=cf_cfg["alpha"])
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    data = load_processed()
    train = validate_interactions(data["train"])
    test = validate_interactions(data["test"])
    items = data["items"]

    weights = build_weighted_counts(train)
    matrix, user_index, item_index = build_implicit_matrix(train, weights)
    # implicit-feedback confidence: c_ui = 1 + alpha * r_ui
    matrix = matrix.copy()
    matrix.data = 1.0 + args.alpha * matrix.data

    model = AlternatingLeastSquares(
        factors=args.factors,
        iterations=args.iterations,
        regularization=cf_cfg["regularization"],
        num_threads=0,
        random_state=42,
    )
    print(f"[cf] fitting ALS on {matrix.shape[0]} users x {matrix.shape[1]} items ...")
    model.fit(matrix)

    item_index_inv = {col: item for item, col in item_index.items()}

    artifact = ModelArtifact(
        name="cf_als",
        version=args.version,
        hyperparameters={
            "factors": args.factors,
            "iterations": args.iterations,
            "alpha": args.alpha,
            "regularization": cf_cfg["regularization"],
        },
        metrics={
            "n_train_interactions": int(len(train)),
            "n_users": len(user_index),
            "n_items": len(item_index),
        },
    )
    artifact.save()
    np.savez(
        artifact.dir / "als_vectors.npz",
        user_factors=model.user_factors,
        item_factors=model.item_factors,
    )
    user_index_df = pd.DataFrame(sorted(user_index.items()), columns=["user_id", "row"])
    item_index_df = pd.DataFrame(sorted(item_index.items()), columns=["item_id", "col"])
    user_index_df.to_parquet(artifact.dir / "user_index.parquet", index=False)
    item_index_df.to_parquet(artifact.dir / "item_index.parquet", index=False)
    add_to_registry(artifact)

    if args.eval_users > 0:
        train_items = set(train["item_id"])
        truth = build_ground_truth(test, train, catalog=train_items)
        users = select_eval_users(
            truth, args.eval_users, restrict=set(train["user_id"])
        )
        print(f"[cf] evaluating on {len(users)} users / {len(truth)} with ground truth")

        topk = int(cf_cfg["k"])
        recs = {}
        for user in users:
            row = user_index.get(user)
            if row is None:
                recs[user] = []
                continue
            ids, scores = model.recommend(
                row,
                matrix[row],
                N=topk,
                filter_already_liked_items=True,
            )
            recs[user] = [item_index_inv[i] for i in ids]

        metric_rows = evaluate_ranking_metrics(
            recs, {u: truth[u] for u in users}, eval_cfg["k_values"]
        )
        metric_rows["coverage"] = coverage(recs, train_items)
        item_to_category = dict(zip(items["item_id"], items["category"], strict=False))
        metric_rows["diversity"] = diversity_category(recs, item_to_category, topk)
        metric_rows["n_eval_users"] = float(len(users))
        print("[cf] " + summarize(metric_rows))

        record_experiment(
            model="cf_als",
            version=args.version,
            dataset=dataset_version(),
            metrics=metric_rows,
            notes=f"implicit ALS, confidence=1+{args.alpha}*r, factors={args.factors}",
        )
    else:
        record_experiment(
            model="cf_als", version=args.version, dataset=dataset_version(), metrics={}
        )


if __name__ == "__main__":
    main()
