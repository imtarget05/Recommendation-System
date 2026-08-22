"""Production hardening regression tests.

Covers the three production incidents:
  #2 `No module named 'sentence_transformers'`  -> import-boundary test
  #3 `'NoneType' object has no attribute 'encode'` -> encoder None-safe tests
  IndexError (user 5889 vs 453)                 -> artifact integrity fail-fast
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_mod
from api.main import ArtifactIntegrityError, NumpyRetriever

# ── helpers ──────────────────────────────────────────────────────────────

def _make_retriever(n_users: int, n_items: int, dim: int = 8, seed: int = 0) -> NumpyRetriever:
    rng = np.random.default_rng(seed)
    return NumpyRetriever(
        user_emb=rng.normal(size=(n_users, dim)).astype(np.float32),
        item_emb=rng.normal(size=(n_items, dim)).astype(np.float32),
        user_ids=np.array([f"ml_user_{i}" for i in range(n_users)], dtype=np.str_),
        item_ids=np.array([f"ml_{i}" for i in range(n_items)], dtype=np.str_),
        version="test",
    )


def _state_with(retriever: NumpyRetriever):
    """AppState whose dataset maps match the given retriever."""
    state = api_mod.AppState()
    items = pd.DataFrame({
        "item_id": list(map(str, retriever.item_ids)),
        "title": [f"T{i}" for i in range(len(retriever.item_ids))],
        "category": ["X"] * len(retriever.item_ids),
        "tags": ["t"] * len(retriever.item_ids),
    })
    train = pd.DataFrame({
        "user_id": list(map(str, retriever.user_ids)),
        "item_id": [str(retriever.item_ids[0])] * len(retriever.user_ids),
        "event_type": ["view"] * len(retriever.user_ids),
        "rating": [4.0] * len(retriever.user_ids),
    })
    state.items_df = items
    state.train_df = train
    state.item_id_map = {iid: i for i, iid in enumerate(items["item_id"])}
    state.idx_to_item = {v: k for k, v in state.item_id_map.items()}
    state.n_items = len(items)
    state.user_id_map = {uid: i for i, uid in enumerate(sorted(train["user_id"].unique()))}
    state.model = retriever
    return state


# ── PHASE 2: artifact/dataset mismatch must fail fast ───────────────────

def test_mismatch_more_dataset_users_than_model_fails_fast(monkeypatch) -> None:
    """Bug replay: model has 453 users, dataset has 5889 -> startup FAIL, not IndexError."""
    model = _make_retriever(n_users=453, n_items=100)
    state = _state_with(model)
    extra = pd.DataFrame({
        "user_id": [f"ml_user_{i}" for i in range(5889)],
        "item_id": ["ml_0"] * 5889,
        "event_type": ["view"] * 5889,
        "rating": [4.0] * 5889,
    })
    state.train_df = extra
    state.user_id_map = {uid: i for i, uid in enumerate(sorted(extra["user_id"].unique()))}
    monkeypatch.setattr(api_mod, "state", state)
    with pytest.raises(ArtifactIntegrityError, match="dataset users"):
        api_mod._validate_artifacts(model)


def test_mismatch_item_count_fails_fast(monkeypatch) -> None:
    """Item embedding count != items dataset count -> startup FAIL."""
    model = _make_retriever(n_users=10, n_items=50)
    state = _state_with(model)
    state.n_items = 51  # one extra catalog row
    monkeypatch.setattr(api_mod, "state", state)
    with pytest.raises(ArtifactIntegrityError, match="dataset items"):
        api_mod._validate_artifacts(model)


def test_mismatch_user_count_vs_emb_rows_fails_fast(monkeypatch) -> None:
    model = _make_retriever(n_users=10, n_items=5)
    state = _state_with(model)
    bad = NumpyRetriever(
        user_emb=model.user_emb[:-1],  # one row short
        item_emb=model.item_emb,
        user_ids=model.user_ids,
        item_ids=model.item_ids,
    )
    state.model = bad
    monkeypatch.setattr(api_mod, "state", state)
    with pytest.raises(ArtifactIntegrityError, match="user_ids length"):
        api_mod._validate_artifacts(bad)


def test_duplicate_item_ids_fail(monkeypatch) -> None:
    model = _make_retriever(n_users=3, n_items=5)
    state = _state_with(model)
    dup = model.item_ids.copy()
    dup[-1] = dup[0]
    bad = NumpyRetriever(model.user_emb, model.item_emb, model.user_ids, dup)
    state.model = bad
    monkeypatch.setattr(api_mod, "state", state)
    with pytest.raises(ArtifactIntegrityError, match="unique item ids"):
        api_mod._validate_artifacts(bad)


def test_nonfinite_embeddings_fail(monkeypatch) -> None:
    model = _make_retriever(n_users=3, n_items=5)
    state = _state_with(model)
    model.item_emb[0, 0] = np.nan
    state.model = model
    monkeypatch.setattr(api_mod, "state", state)
    with pytest.raises(ArtifactIntegrityError, match="finite"):
        api_mod._validate_artifacts(model)


def test_matching_artifacts_pass(monkeypatch, tmp_path) -> None:
    # Point MODEL_PATH at a location with no manifest so the (real) production
    # manifest for the 453-user release is not compared against this 10-user toy.
    monkeypatch.setattr(api_mod, "MODEL_PATH", str(tmp_path / "embeddings.npz"))
    model = _make_retriever(n_users=10, n_items=20)
    monkeypatch.setattr(api_mod, "state", _state_with(model))
    api_mod._validate_artifacts(model)  # must not raise


def test_no_assert_in_validation_source() -> None:
    """Validation must use explicit exceptions, not `assert` (stripped under -O)."""
    src = open(api_mod.__file__, encoding="utf-8").read()
    fn_start = src.index("def _validate_artifacts")
    body = src[fn_start:src.index("\nclass ", fn_start)]
    assert " assert " not in body


# ── PHASE 3: production API must not require torch / sentence_transformers ──

def test_api_importable_without_torch_or_sentence_transformers() -> None:
    """Real import-boundary test: hard-block training-only modules in a fresh
    interpreter, then import the API module. Bug #2 replay."""
    script = (
        "import sys\n"
        "for m in ('torch', 'sentence_transformers', 'implicit'):\n"
        "    sys.modules[m] = None  # hard-block even indirect imports\n"
        "import api.main\n"
        "print('IMPORT_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, (
        f"API import failed without training deps:\n{result.stderr[-1500:]}"
    )
    assert "IMPORT_OK" in result.stdout


# ── PHASE 4: encoder None-safe / fallback behavior ───────────────────────

@pytest.fixture()
def search_client(monkeypatch):
    state = _state_with(_make_retriever(3, 6))
    monkeypatch.setattr(api_mod, "state", state)
    return TestClient(api_mod.app)


def test_search_encoder_none_falls_back_to_keyword(search_client: TestClient) -> None:
    """Bug #3 replay: encoder is None -> must NOT crash, fall back to keyword."""
    assert api_mod.state.embedder is None
    resp = search_client.get("/search?q=T1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "keyword"
    assert body["items"], "keyword fallback should return results"


def test_search_encoder_error_falls_back_not_500(search_client: TestClient, monkeypatch) -> None:
    """Case C: encoder init/encode raises -> keyword fallback, still HTTP 200."""

    def boom():
        raise RuntimeError("onnx download failed")

    monkeypatch.setattr(api_mod, "_get_embedder", boom)
    resp = search_client.get("/search?q=T2")
    assert resp.status_code == 200
    assert resp.json()["source"] == "keyword"


def test_search_semantic_when_qdrant_and_encoder_available(
    search_client: TestClient, monkeypatch
) -> None:
    """Case B: healthy encoder + Qdrant -> source == semantic."""

    class FakeEncoder:
        def encode(self, texts, convert_to_numpy=False):
            return np.eye(len(texts), 384, dtype=np.float32)

    class FakeHit:
        payload = {"item_id": "ml_1"}
        score = 0.97

    import training.embeddings as emb_mod

    monkeypatch.setattr(api_mod, "_get_embedder", lambda: FakeEncoder())
    api_mod.state.qdrant_client = object()  # any non-None client
    api_mod.state.qdrant_available = True
    monkeypatch.setattr(emb_mod, "query_qdrant", lambda *a, **kw: [FakeHit()])

    resp = search_client.get("/search?q=anything")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "semantic"
    assert body["items"][0]["item_id"] == "ml_1"

