"""Ingest CLI: download -> load -> normalize -> validate -> split -> write parquet + manifest.

Usage:
    uv run python -m data.scripts.ingest --dataset movielens --version ml-latest-small
    uv run python -m data.scripts.ingest --dataset hm
    uv run python -m data.scripts.ingest --dataset huggingface --version <repo_id>
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from app.config import load_config
from data.connectors.base import get_loader
from data.preprocessing.normalize import (
    filter_low_activity_users,
    trim_to_max_rows,
    validate_interactions,
    validate_items,
    validate_users,
)
from data.preprocessing.split import split_by_user_history, split_interactions_time_based

from .manifest import DatasetManifest, ManifestDataset, sha256_of


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Ingest raw datasets into the internal schema")
    parser.add_argument(
        "--dataset",
        default=cfg.data.default_dataset,
        choices=["movielens", "hm", "huggingface"],
    )
    parser.add_argument(
        "--version",
        default=None,
        help="dataset version (e.g. ml-latest-small, ml-25m, HF repo id)",
    )
    parser.add_argument("--max-rows", type=int, default=cfg.data.max_rows)
    parser.add_argument(
        "--min-interactions", type=int, default=cfg.data.min_interactions_per_user
    )
    parser.add_argument("--split-mode", choices=["global", "per-user"], default="global")
    parser.add_argument(
        "--no-limit", action="store_true", help="ignore max_rows (use full dataset)"
    )
    args = parser.parse_args()

    raw_dir = cfg.data.raw_dir
    processed = cfg.data.processed_dir
    manifests_dir = cfg.data.manifests_dir

    loader = get_loader(args.dataset, version=args.version)

    print(f"[ingest] dataset={args.dataset} version={loader.version} max_rows={args.max_rows}")
    loaded = loader.load(raw_dir)
    meta = loaded.meta
    print(f"[ingest] source_path={loaded.source_path}")
    print(f"[ingest] meta={json.dumps(meta, indent=2, default=str)}")

    interactions = validate_interactions(loaded.interactions)
    items = validate_items(loaded.items)
    users = validate_users(loaded.users)

    if not args.no_limit:
        interactions = trim_to_max_rows(interactions, args.max_rows)
    interactions = filter_low_activity_users(interactions, args.min_interactions)

    if args.split_mode == "global":
        result = split_interactions_time_based(interactions)
    else:
        result = split_by_user_history(interactions, min_history=args.min_interactions)

    processed.mkdir(parents=True, exist_ok=True)
    for name, subset in (
        ("train", result.train),
        ("validation", result.validation),
        ("test", result.test),
    ):
        subset.to_parquet(processed / f"interactions_{name}.parquet", index=False)
    items.to_parquet(processed / "items.parquet", index=False)
    users.to_parquet(processed / "users.parquet", index=False)

    checksum = sha256_of(loaded.source_path) if loaded.source_path else None
    manifest = DatasetManifest(
        dataset=ManifestDataset(
            name=args.dataset,
            source=meta.get("source", args.dataset),
            version=loader.version,
            row_count=len(interactions),
            downloaded_at=datetime.now(UTC),
            checksum=checksum,
        ),
        schema_fields={
            "interactions": list(interactions.columns),
            "items": list(items.columns),
            "users": list(users.columns),
        },
        splits={
            "train": len(result.train),
            "validation": len(result.validation),
            "test": len(result.test),
        },
    )
    manifest.save(manifests_dir / "dataset_manifest.yaml")

    print("\n=== MANIFEST ===")
    print((manifests_dir / "dataset_manifest.yaml").read_text())

    print("\n=== SPLIT SUMMARY ===")
    for name, subset in (
        ("train", result.train),
        ("validation", result.validation),
        ("test", result.test),
    ):
        tmin, tmax = subset["timestamp"].min(), subset["timestamp"].max()
        print(f"{name:12s} rows={len(subset):>10,d}  users={subset['user_id'].nunique():>9,d}  "
              f"span={tmin:%Y-%m-%d} -> {tmax:%Y-%m-%d}")
    thr = result.thresholds
    if thr and thr["ratios"]:
        print("time thresholds:", {k: (str(v) if v else None) for k, v in thr.items()})

    print("\n=== SAMPLE INTERACTIONS ===")
    print(interactions.sample(min(5, len(interactions)), random_state=42).to_string())
    print("\n=== SAMPLE ITEMS ===")
    print(items.sample(min(5, len(items)), random_state=42).to_string())
    print("\n=== EVENT DISTRIBUTION ===")
    print(interactions["event_type"].value_counts().to_string())
    print("\n[ingest] DONE")


if __name__ == "__main__":
    main()
