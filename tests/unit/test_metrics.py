"""P3 observability metrics tests: search outcome counters, ONNX/Qdrant instrumentation.

Cases:
A. semantic success  -> semantic_total +1, fallback unchanged
B. encoder unavailable -> fallback +1, source="keyword"
C. Qdrant unavailable at request time (client None) -> fallback +1, no qdrant_requests
D. repeated requests -> exact increment counts (no double-count)
E. concurrent requests -> totals match request count
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

import api.main as api_mod


class _FakeEncoder:
    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> Any:
        import numpy as np

        return np.ones((len(texts), 384), dtype=np.float32)


class _FakeHit:
    def __init__(self) -> None:
        self.payload = {"item_id": "ml_1"}
        self.score = 0.9


def _make_client(monkeypatch, *, encoder: Any, qdrant: Any) -> TestClient:
    state = api_mod.AppState()
    state.items_df = pd.DataFrame({
        "item_id": ["ml_1", "ml_2"],
        "title": ["Toy Story", "Braveheart"],
        "category": ["Animation", "Drama"],
        "tags": ["toy", "war"],
    })
    state.train_df = pd.DataFrame({
        "user_id": ["u1"], "item_id": ["ml_1"],
        "event_type": ["view"], "rating": [5.0],
    })
    state.item_id_map = {"ml_1": 0, "ml_2": 1}
    state.idx_to_item = {0: "ml_1", 1: "ml_2"}
    state.n_items = 2
    state.user_id_map = {"u1": 0}
    state.embedder = encoder
    state.qdrant_client = qdrant
    state.qdrant_available = qdrant is not None
    monkeypatch.setattr(api_mod, "state", state)
    # Reset the counters under test so each test starts from a known base.
    for c in (
        api_mod.SEMANTIC_SEARCH_TOTAL,
        api_mod.KEYWORD_FALLBACK_TOTAL,
        api_mod.QDRANT_REQUESTS,
    ):
        c.clear()
        c.labels(*[{} for _ in c._labelnames]).inc(0) if c._labelnames else c.inc(0)
    api_mod.QDRANT_ERRORS.clear()
    if api_mod.QDRANT_ERRORS._labelnames:
        for et in api_mod._ERROR_TYPES:
            api_mod.QDRANT_ERRORS.labels(error_type=et).inc(0)
    return TestClient(api_mod.app)


def _counter_value(metric_family: str, text: str, **labels: str) -> float:
    """Read a counter value from Prometheus text exposition."""
    name = metric_family + "_total" if not metric_family.endswith("_total") else metric_family
    label_str = "".join(f'{k}="{v}"' for k, v in labels.items())
    pat = re.compile(rf"^{name}(?:\{{{label_str}\}})?\s+([0-9.e+-]+)$", re.M)
    m = pat.search(text)
    return float(m.group(1)) if m else 0.0


def _snapshot(client: TestClient, *names: str, **labels: str) -> dict[str, float]:
    """Read current values of metric families (missing family => 0.0)."""
    text = client.get("/metrics").text
    return {n: _counter_value(n, text, **labels) for n in names}


def test_semantic_success_increments_semantic_counter(monkeypatch) -> None:
    class _Qdrant:
        def query_points(self, **kwargs: Any) -> list[Any]:
            return [_FakeHit()]

    client = _make_client(monkeypatch, encoder=_FakeEncoder(), qdrant=_Qdrant())
    before = _snapshot(
        client,
        "recsys_semantic_search_total",
        "recsys_keyword_fallback_total",
        "recsys_qdrant_requests_total",
    )
    r = client.get("/search?q=toy")
    assert r.status_code == 200 and r.json()["source"] == "semantic"
    metrics = client.get("/metrics").text
    assert _counter_value("recsys_semantic_search_total", metrics) - before[
        "recsys_semantic_search_total"
    ] == 1.0
    assert _counter_value("recsys_keyword_fallback_total", metrics) == before[
        "recsys_keyword_fallback_total"
    ]
    assert _counter_value("recsys_qdrant_requests_total", metrics) - before[
        "recsys_qdrant_requests_total"
    ] == 1.0
    assert "recsys_onnx_inference_seconds" in metrics
    assert "recsys_qdrant_latency_seconds" in metrics


def test_encoder_none_falls_back_and_counts(monkeypatch) -> None:
    class _Qdrant:
        def query_points(self, **kwargs: Any) -> list[Any]:
            return [_FakeHit()]

    client = _make_client(monkeypatch, encoder=None, qdrant=_Qdrant())
    def _boom():
        raise RuntimeError("no encoder")

    monkeypatch.setattr(api_mod, "_get_embedder", _boom)
    before = _snapshot(
        client, "recsys_semantic_search_total", "recsys_keyword_fallback_total"
    )
    r = client.get("/search?q=toy")
    assert r.status_code == 200 and r.json()["source"] == "keyword"
    metrics = client.get("/metrics").text
    assert _counter_value("recsys_keyword_fallback_total", metrics) - before[
        "recsys_keyword_fallback_total"
    ] == 1.0
    assert _counter_value("recsys_semantic_search_total", metrics) == before[
        "recsys_semantic_search_total"
    ]


def test_no_qdrant_client_fallback_only(monkeypatch) -> None:
    client = _make_client(monkeypatch, encoder=_FakeEncoder(), qdrant=None)
    before = _snapshot(
        client,
        "recsys_keyword_fallback_total",
        "recsys_qdrant_requests_total",
        "recsys_qdrant_errors_total",
        error_type="other",
    )
    r = client.get("/search?q=toy")
    assert r.status_code == 200 and r.json()["source"] == "keyword"
    metrics = client.get("/metrics").text
    assert _counter_value("recsys_keyword_fallback_total", metrics) - before[
        "recsys_keyword_fallback_total"
    ] == 1.0
    # No Qdrant call was attempted.
    assert _counter_value("recsys_qdrant_requests_total", metrics) == before[
        "recsys_qdrant_requests_total"
    ]
    assert _counter_value(
        "recsys_qdrant_errors_total", metrics, error_type="other"
    ) == before["recsys_qdrant_errors_total"]


def test_repeated_requests_exact_counts(monkeypatch) -> None:
    class _Qdrant:
        def query_points(self, **kwargs: Any) -> list[Any]:
            return [_FakeHit()]

    client = _make_client(monkeypatch, encoder=_FakeEncoder(), qdrant=_Qdrant())
    before = _snapshot(
        client, "recsys_semantic_search_total", "recsys_qdrant_requests_total"
    )
    for _ in range(5):
        assert client.get("/search?q=toy").json()["source"] == "semantic"
    metrics = client.get("/metrics").text
    assert _counter_value("recsys_semantic_search_total", metrics) - before[
        "recsys_semantic_search_total"
    ] == 5.0
    assert _counter_value("recsys_qdrant_requests_total", metrics) - before[
        "recsys_qdrant_requests_total"
    ] == 5.0


def test_concurrent_requests_no_double_count(monkeypatch) -> None:
    import concurrent.futures

    class _Qdrant:
        def query_points(self, **kwargs: Any) -> list[Any]:
            return [_FakeHit()]

    client = _make_client(monkeypatch, encoder=_FakeEncoder(), qdrant=_Qdrant())
    before = _snapshot(client, "recsys_semantic_search_total")

    def hit(_: int) -> str:
        return client.get("/search?q=toy").json()["source"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        sources = list(ex.map(hit, range(20)))
    assert all(s == "semantic" for s in sources)
    metrics = client.get("/metrics").text
    assert _counter_value("recsys_semantic_search_total", metrics) - before[
        "recsys_semantic_search_total"
    ] == 20.0
