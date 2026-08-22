"""Hybrid /chat/recommend regression tests (message-aware retrieval + fusion).

Covers: known-user intent sensitivity, unknown-user semantic fallback,
Vietnamese messages, generic-message personal dominance, contract stability.
Uses FastAPI TestClient + monkeypatched state/candidates — no real Qdrant/model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as api_mod


def _item(iid: str, title: str, cat: str, score: float):
    return api_mod.RecItem(item_id=iid, title=title, category=cat, score=score)


SEM_ACTION = [_item("a1", "Steel", "Action", 0.9), _item("a2", "Heat", "Action", 0.8)]
SEM_ROMANCE = [_item("r1", "Before Sunrise", "Romance", 0.9),
               _item("r2", "Notting Hill", "Romance", 0.8)]
PERSONAL_429 = [_item("p1", "Copycat", "Crime", 0.98),
                _item("p2", "Jungle Book", "Adventure", 0.97)]
PERSONAL_100 = [_item("q1", "Toy Story", "Children", 0.95),
                _item("q2", "Aladdin", "Animation", 0.9)]


@pytest.fixture()
def client(monkeypatch):
    """TestClient with a blanked-out state (no real Qdrant/embedder/model)."""
    state = api_mod.state
    for attr in ("qdrant_client", "qdrant_collection", "item_id_map",
                 "user_id_map", "llm", "items_df", "model"):
        if hasattr(state, attr):
            try:
                setattr(state, attr, None)
            except Exception:  # noqa: BLE001
                pass
    if hasattr(state, "qdrant_available"):
        state.qdrant_available = False
    if hasattr(state, "user_id_map"):
        state.user_id_map = {}
    return TestClient(api_mod.app)


def _patch_sem(monkeypatch, mapping):
    """Patch _chat_semantic_candidates with per-substring canned candidates."""
    def fake(message: str):
        low = message.lower()
        for key, (items, cats) in mapping.items():
            if key in low:
                return items, cats, True
        return [], [], False
    monkeypatch.setattr(api_mod, "_chat_semantic_candidates", fake)


# ── Contract ─────────────────────────────────────────────────

def test_contract_fields_preserved(client):
    r = client.post("/chat/recommend", json={
        "user_id": "ml_user_x", "message": "phim hành động", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    for key in ("user_id", "message", "items", "latency_ms", "llm_mode",
                "prompt_version"):
        assert key in body


# ── Known user: intent sensitivity ───────────────────────────

def test_known_user_action_differs_romance(client, monkeypatch):
    _patch_sem(monkeypatch, {"action": (SEM_ACTION, ["Action"]),
                             "romance": (SEM_ROMANCE, ["Romance"])})
    monkeypatch.setattr(api_mod.state, "user_id_map", {"ml_user_429": 0})
    monkeypatch.setattr(api_mod, "_run_engine", lambda uid, k: PERSONAL_429[:k])
    a = client.post("/chat/recommend", json={
        "user_id": "ml_user_429", "message": "action movie", "top_k": 5}).json()
    r = client.post("/chat/recommend", json={
        "user_id": "ml_user_429", "message": "romance movie", "top_k": 5}).json()
    ids_a = [i["item_id"] for i in a["items"]]
    ids_r = [i["item_id"] for i in r["items"]]
    assert ids_a and ids_r and ids_a != ids_r
    assert ids_a[0] == "a1"          # semantic top-1 leads under strong intent
    assert ids_r[0] == "r1"


def test_same_message_personalized_differently(client, monkeypatch):
    _patch_sem(monkeypatch, {"action": (SEM_ACTION, ["Action"])})
    monkeypatch.setattr(api_mod.state, "user_id_map",
                        {"ml_user_429": 0, "ml_user_100": 1})

    def engine(uid, k):
        return (PERSONAL_429 if uid == 0 else PERSONAL_100)[:k]
    monkeypatch.setattr(api_mod, "_run_engine", engine)

    a = client.post("/chat/recommend", json={
        "user_id": "ml_user_429", "message": "action movies", "top_k": 6}).json()
    b = client.post("/chat/recommend", json={
        "user_id": "ml_user_100", "message": "action movies", "top_k": 6}).json()
    ids_a = [i["item_id"] for i in a["items"]]
    ids_b = [i["item_id"] for i in b["items"]]
    assert "a1" in ids_a and "a2" in ids_a      # both get Action intent
    assert ("p1" in ids_a or "p2" in ids_a)     # personal items present
    assert ("q1" in ids_b or "q2" in ids_b)     # different personalization
    assert set(ids_a) != set(ids_b)


# ── Unknown user ─────────────────────────────────────────────

def test_unknown_user_semantic_only(client, monkeypatch):
    _patch_sem(monkeypatch, {"action": (SEM_ACTION, ["Action"])})
    body = client.post("/chat/recommend", json={
        "user_id": "ml_user_999999", "message": "action movie", "top_k": 3}).json()
    assert len(body["items"]) == 2
    assert all(i["category"] == "Action" for i in body["items"])


# ── Vietnamese ───────────────────────────────────────────────

def test_vietnamese_message_maps_to_intent(client, monkeypatch):
    seen: dict[str, tuple] = {}

    def fake(message: str):
        from app.query_norm import normalize_query
        norm = normalize_query(message)
        seen[message] = (norm.normalized, norm.categories)
        if norm.categories == ["Action"]:
            return SEM_ACTION, ["Action"], True
        return [], [], False
    monkeypatch.setattr(api_mod, "_chat_semantic_candidates", fake)

    body = client.post("/chat/recommend", json={
        "user_id": "ml_user_999999",
        "message": "tôi muốn xem phim hành động", "top_k": 3}).json()
    assert any(i["category"] == "Action" for i in body["items"])
    norm_msg = list(seen.values())[0][0]
    assert "action" in norm_msg


# ── Generic / ambiguous message ──────────────────────────────

def test_generic_message_personal_dominant(client, monkeypatch):
    _patch_sem(monkeypatch, {})
    monkeypatch.setattr(api_mod.state, "user_id_map", {"ml_user_429": 0})
    monkeypatch.setattr(api_mod, "_run_engine", lambda uid, k: PERSONAL_429[:k])
    body = client.post("/chat/recommend", json={
        "user_id": "ml_user_429", "message": "recommend something for me",
        "top_k": 3}).json()
    ids = {i["item_id"] for i in body["items"]}
    assert ids <= {"p1", "p2"} and ids


def test_empty_candidates_no_crash(client):
    body = client.post("/chat/recommend", json={
        "user_id": "ml_user_x", "message": "zzz", "top_k": 3}).json()
    assert isinstance(body["items"], list)


# ── Fusion helpers ───────────────────────────────────────────

def test_minmax_normalization() -> None:
    n = api_mod._minmax([_item("a", "A", "Action", 0.2),
                         _item("b", "B", "Drama", 0.8)])
    assert n["a"] == 0.0 and n["b"] == 1.0


def test_fusion_presets_documented_weights() -> None:
    assert api_mod._FUSION_PRESETS == {"A": 0.70, "B": 0.50, "C": 0.30}

