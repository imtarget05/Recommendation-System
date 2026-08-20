"""Popularity baseline (Section 8.1): rank items by weighted interaction counts.

Usage:
    uv run python -m training.baseline_popularity [--eval-users N] [--recency]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

from data.preprocessing.normalize import recency_weight, validate_interactions
from training.common import (
    ModelArtifact,
    add_to_registry,
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
    pop_cfg = cfg["popularity"]
    eval_cfg = cfg["eval"]

    parser = argparse.ArgumentParser(description="Train + evaluate the popularity baseline")
    parser.add_argument("--eval-users", type=int, default=eval_cfg["max_users"])
    parser.add_argument(
        "--recency",
        action="store_true",
        default=pop_cfg.get("recency_weighted", False),
    )
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    data = load_processed()
    train = validate_interactions(data["train"])
    test = validate_interactions(data["test"])
    items = data["items"]

    weights = build_weighted_counts(train)
    if args.recency:
        weights = weights * recency_weight(cast(pd.Series, train["timestamp"]))

    grouped = pd.DataFrame({"item_id": train["item_id"], "w": weights}).groupby("item_id")[
        "w"
    ].sum()
    scores: pd.Series = cast(pd.Series, grouped).sort_values(ascending=False)

    topk = int(pop_cfg["k"])
    item_to_category = dict(zip(items["item_id"], items["category"], strict=False))

    artifact = ModelArtifact(
        name="popularity",
        version=args.version,
        hyperparameters={"k": topk, "recency_weighted": args.recency},
        metrics={"n_train_interactions": int(len(train)), "n_items": int(len(scores))},
    )
    artifact.save()
    scores.rename("score").to_frame().to_parquet(artifact.dir / "popularity.parquet")
    add_to_registry(artifact)

    print(f"[pop] top item {scores.index[0]} score={float(scores.max()):.3f}")

    if args.eval_users > 0:
        train_items = set(train["item_id"])
        truth = build_ground_truth(test, train, catalog=train_items)
        users = select_eval_users(truth, args.eval_users, restrict=set(train["user_id"]))
        print(f"[pop] evaluating on {len(users)} users / {len(truth)} with ground truth")

        recs = {user: [str(x) for x in scores.index.tolist()[:topk]] for user in users}
        metric_rows = evaluate_ranking_metrics(
            recs, {u: truth[u] for u in users}, eval_cfg["k_values"]
        )
        metric_rows["coverage"] = coverage(recs, train_items)
        metric_rows["diversity"] = diversity_category(recs, item_to_category, topk)
        metric_rows["n_eval_users"] = float(len(users))
        print("[pop] " + summarize(metric_rows))

        record_experiment(
            model="popularity",
            version=args.version,
            dataset=dataset_version(),
            metrics=metric_rows,
            notes="pure weighted-count popularity baseline (no recency)"
            if not args.recency
            else "recency-weighted popularity",
        )
    else:
        record_experiment(
            model="popularity", version=args.version, dataset=dataset_version(), metrics={}
        )


if __name__ == "__main__":
    main()
