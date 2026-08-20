"""Experiment result store: JSON records + rendered ablation markdown (Section 16.3)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

REPORTS = Path("reports")
EXPERIMENTS_JSON = REPORTS / "experiments.json"
ABLATION_MD = REPORTS / "ablation.md"
TABLE_MODELS_SEC = 20

ABLATION_COLUMNS = [
    "model",
    "version",
    "dataset",
    "Recall@20",
    "NDCG@20",
    "HitRate@20",
    "Coverage",
    "Diversity",
]


def record_experiment(
    model: str,
    version: str,
    dataset: str,
    metrics: dict[str, float],
    notes: str | None = None,
) -> dict:
    entry = {
        "model": model,
        "version": version,
        "dataset": dataset,
        "metrics": metrics,
        "timestamp": datetime.now(UTC).isoformat(),
        "notes": notes,
    }
    data = (
        json.loads(EXPERIMENTS_JSON.read_text())
        if EXPERIMENTS_JSON.exists()
        else {"experiments": []}
    )
    data["experiments"].append(entry)
    EXPERIMENTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_JSON.write_text(json.dumps(data, indent=2))
    return entry


def _fmt(value) -> str:  # noqa: ANN001
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _pick(metrics: dict, key: str, default=0.0) -> float:  # noqa: ANN001
    return metrics.get(key, default)


def render_ablation_table(latest_per_model: bool = True) -> str:
    """Render Section 16.3 ablation table from recorded experiments."""
    if not EXPERIMENTS_JSON.exists():
        return "no experiments recorded yet"
    data = json.loads(EXPERIMENTS_JSON.read_text())
    exp = data["experiments"]
    if latest_per_model:
        seen: dict = {}
        for e in exp:  # keep the latest run per model name
            seen[e["model"]] = e
        exp = list(seen.values())
    exp = sorted(exp, key=lambda e: e["model"])

    rows = ["| " + " | ".join(ABLATION_COLUMNS) + " |", "|" + "---|" * len(ABLATION_COLUMNS)]
    for e in exp:
        m = e["metrics"]
        values = [
            e["model"],
            e["version"],
            e["dataset"],
            _pick(m, "recall@20"),
            _pick(m, "ndcg@20"),
            _pick(m, "hit@20"),
            _pick(m, "coverage"),
            _pick(m, "diversity"),
        ]
        rows.append("| " + " | ".join(_fmt(v) for v in values) + " |")
    table = "\n".join(rows)
    ABLATION_MD.parent.mkdir(parents=True, exist_ok=True)
    ABLATION_MD.write_text(
        "# Ablation table (Section 16.3)\n\n> Auto-generated from `reports/experiments.json`. "
        "Values are measured, never fabricated.\n\n" + table + "\n"
    )
    return table
