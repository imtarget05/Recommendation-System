"""LLMOps drift monitor CLI — run with: `make monitor-drift` (Section 10A.5).

Samples either real inference scores from data/embeddings when the pipeline has run,
or deterministic synthetic data (seeded) so the command always demonstrates the drift
metric end-to-end. Prints a compact PSI/alert report per monitored column.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from monitoring.drift import detect_drift

EMBEDDINGS_DIR = Path("data/embeddings")
REFERENCE_FILE = EMBEDDINGS_DIR / "reference_scores.npy"
CURRENT_FILE = EMBEDDINGS_DIR / "current_scores.npy"


def _load_or_synthetic() -> tuple[np.ndarray, np.ndarray]:
    """Real score distributions if present, else deterministic synthetic ones."""
    if REFERENCE_FILE.exists() and CURRENT_FILE.exists():
        ref = np.load(REFERENCE_FILE)
        cur = np.load(CURRENT_FILE)
        return ref, cur
    rng = np.random.default_rng(10)  # deterministic, reproducible
    ref = rng.normal(loc=0.0, scale=0.05, size=2000)
    cur = rng.normal(loc=0.2, scale=0.05, size=2000)  # injected drift
    return ref, cur


def main() -> int:
    from app.config import load_config

    ref, cur = _load_or_synthetic()
    ref_df = {"score": ref}
    cur_df = {"score": cur}
    res = detect_drift(
        ref_df, cur_df, numeric_columns=["score"], categorical_columns=[]
    )
    threshold = load_config().monitoring.drift_threshold
    score = res["score"]
    alerts = [k for k, v in res.items() if v.get("alert")]
    print("LLMOps drift monitor (10A.5)")
    print(f"  drift_threshold       : {threshold}")
    print(f"  score.psi             : {score['psi']:.4f}")
    print(f"  score.kl_divergence   : {score.get('kl', float('nan')):.4f}")
    print(f"  score.alert           : {score['alert']}")
    print(f"  total alerts          : {len(alerts)}")
    if not alerts:
        print("  status: OK")
        return 0
    print("  status: DRIFT DETECTED -> investigate input distribution shift")
    return 1


if __name__ == "__main__":
    sys.exit(main())
