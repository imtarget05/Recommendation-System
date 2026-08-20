from __future__ import annotations

import json

import pandas as pd

from training.common import (
    ModelArtifact,
    add_to_registry,
    build_implicit_matrix,
    build_weighted_counts,
    dataset_version,
    load_processed,
)
from training.results import EXPERIMENTS_JSON, record_experiment, render_ablation_table


def test_model_artifact_save_and_metadata(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    artifact = ModelArtifact(
        name="test_model",
        version="v0",
        metrics={"recall@20": 0.123},
        hyperparameters={"factors": 8},
    )
    out = artifact.save()
    assert out.exists()
    meta = (out / "metadata.yaml").read_text()
    assert "model_name: test_model" in meta
    assert "status: candidate" in meta
    assert "dataset_version" in meta


def test_registry_append_and_lifecycle(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    artifact = ModelArtifact("m", "v1", metrics={}, hyperparameters={})
    add_to_registry(artifact)
    entries = json.loads((tmp_path / "models" / "registry.json").read_text())
    assert len(entries) == 1
    assert entries[0]["model_name"] == "m"
    assert entries[0]["status"] == "candidate"


def test_build_implicit_matrix_counts() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2"],
            "item_id": ["a", "b", "a", "a"],
            "event_type": ["view", "click", "like", "view"],
        }
    )
    weights = build_weighted_counts(df)
    matrix, user_index, item_index = build_implicit_matrix(df, weights)
    # u1 has a: 1 + 3 = 4 ; u1 b: 2 ; u2 a: 1
    assert matrix[user_index["u1"], item_index["a"]] == 4.0
    assert matrix[user_index["u1"], item_index["b"]] == 2.0
    assert matrix[user_index["u2"], item_index["a"]] == 1.0


def test_load_processed_missing_raises(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    import pytest

    with pytest.raises(FileNotFoundError, match="missing processed splits"):
        load_processed()


def test_record_experiment_and_render(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    record_experiment(
        model="popularity",
        version="v1",
        dataset="movielens-ml-25m",
        metrics={
            "recall@20": 0.10,
            "ndcg@20": 0.05,
            "hit@20": 0.3,
            "coverage": 0.01,
            "diversity": 0.4,
        },
    )
    record_experiment(
        model="cf_als",
        version="v1",
        dataset="movielens-ml-25m",
        metrics={
            "recall@20": 0.25,
            "ndcg@20": 0.15,
            "hit@20": 0.6,
            "coverage": 0.03,
            "diversity": 0.6,
        },
    )
    assert EXPERIMENTS_JSON.exists()
    table = render_ablation_table()
    assert "popularity" in table
    assert "cf_als" in table
    assert "Recall@20" in table
    assert "0.1000" in table


def test_dataset_version_from_manifest(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import shutil

    from app.config import load_config

    cfg = load_config()
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    manifest = cfg.data.manifests_dir / "dataset_manifest.yaml"
    if manifest.exists():
        shutil.copy(manifest, tmp_path / "data" / "manifests" / "dataset_manifest.yaml")
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    load_config.cache_clear()
    try:
        assert dataset_version().startswith("movielens-ml-")
    finally:
        load_config.cache_clear()
