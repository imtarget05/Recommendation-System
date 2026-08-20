"""Tests for online drift detection (Sections 17, 19.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring.drift import detect_categorical_shift, detect_drift, kl_divergence, psi


def test_psi_stable_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    cur = rng.normal(size=5000)
    assert psi(cur, ref) < 0.01


def test_psi_shifted_is_large() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(size=5000)
    cur = rng.normal(loc=3.0, size=5000)
    assert psi(cur, ref) > 0.1


def test_psi_identical_is_zero() -> None:
    vals = np.arange(100.0)
    assert psi(vals, vals) == 0.0


def test_kl_nonnegative() -> None:
    rng = np.random.default_rng(2)
    ref = rng.normal(size=4000)
    cur = rng.normal(size=4000)
    assert kl_divergence(ref, cur) >= 0.0


def test_detect_drift_numeric_and_categorical() -> None:
    rng = np.random.default_rng(3)
    ref = pd.DataFrame(
        {"score": rng.normal(size=2000), "color": rng.choice(["a", "b", "c"], 2000)}
    )
    cur = pd.DataFrame(
        {"score": rng.normal(loc=2.0, size=2000), "color": rng.choice(["a", "b", "z"], 2000)}
    )
    res = detect_drift(
        ref, cur, numeric_columns=["score"], categorical_columns=["color"]
    )
    assert res["score"]["psi"] > 0.1
    assert res["score"]["alert"] is True
    assert res["color"]["score"] >= 0.0


def test_categorical_shift_handles_new_category() -> None:
    ref = pd.Series(["a", "b", "c"])
    cur = pd.Series(["a", "b", "d"])
    res = detect_categorical_shift(ref, cur)
    assert res["category_deltas"]["d"] > 0.0  # appeared in current
    assert res["category_deltas"]["c"] < 0.0  # dropped from current
    assert res["score"] >= 0.0
