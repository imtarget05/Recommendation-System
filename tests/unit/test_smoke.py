"""Regression tests for the post-deploy smoke test logic (P5, item 15).

Uses httpx.MockTransport — no real network calls. Covers:
  PASS scenario + FAIL A (health 503 exhausted retries) + FAIL B
  (search 200 but source=keyword) + FAIL C (missing metric) +
  FAIL D (malformed JSON) + FAIL E (timeout).
"""

from __future__ import annotations

import httpx
import pytest

from scripts.smoke_test import (
    CHECKS,
    MAX_ATTEMPTS,
    SmokeFailure,
    check_health,
    check_metrics,
    check_search,
    run,
)


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.smoke_test.RETRY_SLEEP", 0)
    monkeypatch.setattr("scripts.smoke_test.REQUEST_TIMEOUT", 2.0)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


GOOD_HEALTH = {"status": "ok", "model_loaded": True, "qdrant_available": True,
               "users": 453, "items": 9742}


def test_pass_scenario() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        p = req.url.path
        if p == "/api/health":
            return httpx.Response(200, json=GOOD_HEALTH)
        if p == "/api/recommend/ml_user_429":
            return httpx.Response(200, json={"items": [
                {"item_id": "ml_1", "title": "T", "category": "C", "score": 0.9}]})
        if p == "/api/search":
            assert req.url.params["q"] == "star"
            return httpx.Response(200, json={"source": "semantic", "items": [
                {"item_id": "ml_2", "score": 0.5}]})
        if p == "/api/metrics":
            names = ("recsys_semantic_search_total recsys_keyword_fallback_total "
                     "recsys_onnx_inference_seconds recsys_qdrant_requests_total "
                     "recsys_qdrant_errors_total recsys_qdrant_latency_seconds")
            return httpx.Response(200, text=names)
        raise AssertionError(p)

    with _client(handler) as c:
        for _, fn in CHECKS:
            fn(c, "http://test")


def test_fail_a_health_transient_exhausts_retries() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    with _client(handler) as c:
        with pytest.raises(SmokeFailure, match="after"):
            check_health(c, "http://test")
    assert len(calls) == MAX_ATTEMPTS  # bounded retry, no storm


def test_fail_b_search_keyword_is_immediate_fail() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"source": "keyword", "items": []})

    with _client(handler) as c:
        with pytest.raises(SmokeFailure, match="source='semantic'.*got source='keyword'"):
            check_search(c, "http://test")
    assert len(calls) == 1  # contract violation must NOT be retried


def test_fail_c_missing_metric() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="recsys_semantic_search_total 1\n")

    with _client(handler) as c:
        with pytest.raises(SmokeFailure, match="metrics missing"):
            check_metrics(c, "http://test")


def test_fail_d_malformed_json() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with _client(handler) as c:
        with pytest.raises(Exception):
            check_health(c, "http://test")


def test_fail_e_timeout_then_success_is_retried() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectTimeout("cold start")
        return httpx.Response(200, json=GOOD_HEALTH)

    with _client(handler) as c:
        check_health(c, "http://test")  # cold-start resilience works
    assert len(calls) == 3


def test_run_reports_failure_exit(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model_loaded": False})  # contract broken

    class FakeClient(httpx.Client):
        pass

    # run() creates its own client; patch the checks' first failure path instead.
    monkeypatch.setattr(
        "scripts.smoke_test.check_health",
        lambda c, b: (_ for _ in ()).throw(SmokeFailure("model_loaded=False")),
    )
    assert run("http://test") is False
    out = capsys.readouterr().out
    assert "SMOKE TEST: FAIL" in out
