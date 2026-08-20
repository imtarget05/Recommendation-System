"""Time-based split (Section 5.5): Past -> TRAIN, Recent -> VALIDATION, Future -> TEST."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.config import load_config


@dataclass
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    thresholds: dict = field(default_factory=dict)


def split_interactions_time_based(
    df: pd.DataFrame, ratios: tuple[float, float, float] | None = None
) -> SplitResult:
    """Split chronologically by global time boundaries (deterministic)."""
    cfg = load_config()
    if ratios is None:
        ratios = (cfg.split.train, cfg.split.validation, cfg.split.test)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")

    ordered = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()

    thresholds = {
        "train_max": train["timestamp"].max() if len(train) else None,
        "val_max": validation["timestamp"].max() if len(validation) else None,
        "test_max": test["timestamp"].max() if len(test) else None,
        "ratios": dict(zip(("train", "validation", "test"), ratios)),
    }
    return SplitResult(train=train, validation=validation, test=test, thresholds=thresholds)


def split_by_user_history(
    df: pd.DataFrame, min_history: int = 2
) -> SplitResult:
    """Per-user temporal split: each user's latest interaction goes to test
    (used for offline evaluation of personalization)."""
    cfg = load_config()
    ratios = (cfg.split.train, cfg.split.validation, cfg.split.test)
    rows = []
    for user_id, group in df.sort_values("timestamp").groupby("user_id", sort=False):
        if len(group) < min_history:
            rows.append(group.assign(split="train"))
            continue
        n = len(group)
        t_end = max(1, int(n * ratios[0]))
        v_end = t_end + max(1, int(n * ratios[1]))
        out = group.copy()
        parts = np.full(n, "train", dtype=object)
        parts[t_end:v_end] = "validation"
        parts[v_end:] = "test"
        out["split"] = parts
        rows.append(out)
    combined = pd.concat(rows, ignore_index=True)
    return SplitResult(
        train=combined[combined["split"] == "train"].drop(columns=["split"]),
        validation=combined[combined["split"] == "validation"].drop(columns=["split"]),
        test=combined[combined["split"] == "test"].drop(columns=["split"]),
    )
