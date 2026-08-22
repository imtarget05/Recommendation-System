"""Serving API tests (Section 14): recommend, event, metrics, fallbacks.

Uses FastAPI TestClient with monkeypatched state — no real data/model load.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_mod
from app.numpy_retriever import NumpyRetriever


def _tiny_retriever(n_users: int, n_items: int, dim: int = 8) -> NumpyRetriever:
    import numpy as np

    rng = np.random.default_rng(7)
    return NumpyRetriever(
        user_emb=rng.standard_normal((n_users, dim)).astype(np.float32),
        item_emb=rng.standard_normal((n_items, dim)).astype(np.float32),
        user_ids=np.array([f"ml_user_{i + 1}" for i in range(n_users)], dtype=np.str_),
        item_ids=np.array([f"ml_{i + 1}" for i in range(n_items)], dtype=np.str_),
    )


@pytest.fixture()
def client(monkeypatch) -> TestClient:  # noqa: ANN001
    """TestClient with a tiny in-memory catalog + model, lifespan skipped."""
    items = pd.DataFrame({
        "item_id": ["ml_1", "ml_2", "ml_3"],
        "title": ["Toy Story", "Braveheart", "Matrix"],
        "category": ["Animation", "Drama", "Sci-Fi"],
        "tags": ["toy", "war", "matrix"],
    })
    train = pd.DataFrame({
        "user_id": ["ml_user_1", "ml_user_1", "ml_user_2"],
        "item_id": ["ml_1", "ml_2", "ml_3"],
        "event_type": ["view", "click", "view"],
        "rating": [5.0, 4.0, 3.0],
    })
    state = api_mod.AppState()
    state.items_df = items
    state.train_df = train
    state.item_id_map = {iid: i for i, iid in enumerate(items["item_id"])}
    state.idx_to_item = {v: k for k, v in state.item_id_map.items()}
    state.n_items = len(items)
    state.user_id_map = {uid: i for i, uid in enumerate(sorted(train["user_id"].unique()))}
    state.model = _tiny_retriever(len(state.user_id_map), state.n_items)
    state.llm = None
    monkeypatch.setattr(api_mod, "state", state)
    return TestClient(api_mod.app)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_recommend_batch_known_user(client: TestClient) -> None:
    resp = client.post("/recommend", json={"user_id": "ml_user_1", "top_n": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "ml_user_1"
    assert body["model_version"] == api_mod.MODEL_VERSION
    assert isinstance(body["items"], list)


def test_recommend_batch_unknown_user(client: TestClient) -> None:
    resp = client.post("/recommend", json={"user_id": "ghost", "top_n": 5})
    assert resp.status_code == 404


def test_recommend_batch_validation(client: TestClient) -> None:
    resp = client.post("/recommend", json={"user_id": "ml_user_1", "top_n": 0})
    assert resp.status_code == 422


def test_event_valid(client: TestClient) -> None:
    resp = client.post("/event", json={
        "user_id": "ml_user_1",
        "item_id": "ml_2",
        "event_type": "click",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["event_type"] == "click"
    assert len(api_mod.state.events_buffer) == 1


def test_event_invalid_type(client: TestClient) -> None:
    resp = client.post("/event", json={
        "user_id": "ml_user_1",
        "item_id": "ml_2",
        "event_type": "swipe",
    })
    assert resp.status_code == 422


def test_event_unknown_item(client: TestClient) -> None:
    resp = client.post("/event", json={
        "user_id": "ml_user_1",
        "item_id": "nope",
        "event_type": "view",
    })
    assert resp.status_code == 404


def test_metrics_expose_prometheus(client: TestClient) -> None:
    client.post("/event", json={
        "user_id": "ml_user_1", "item_id": "ml_3", "event_type": "like",
    })
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "recsys_http_requests_total" in resp.text
    assert "recsys_events_total" in resp.text


def test_popular_fallback(client: TestClient) -> None:
    resp = client.get("/popular?k=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "popularity-baseline"
    assert len(body["items"]) == 2


def test_search_keyword(client: TestClient) -> None:
    resp = client.get("/search?q=toy")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert any("Toy" in i["title"] for i in items)


def test_search_requires_query(client: TestClient) -> None:
    assert client.get("/search").status_code == 422


def test_get_recommend_unknown_user_404(client: TestClient) -> None:
    resp = client.get("/recommend/ghost")
    assert resp.status_code == 404


def test_static_root(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Search" in resp.text or "RecSys" in resp.text
