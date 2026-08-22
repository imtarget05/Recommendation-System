#!/usr/bin/env python3
"""Post-deploy production smoke test (P5).

Validates the LIVE deployment contract end-to-end through the Cloudflare
Worker (default) or directly against Render:

    GET /api/health      -> model_loaded, qdrant_available, users/items > 0
    GET /api/recommend/… -> 200, non-empty numeric-scored items
    GET /api/search?q=.. -> source == "semantic"  (keyword => FAIL)
    GET /api/metrics     -> P3 metrics exposed

Retry policy: ONLY transient failures are retried (502/503/504, timeouts,
connection errors) to survive Render Free cold starts. Contract violations
(e.g. HTTP 200 but source=="keyword") fail IMMEDIATELY — no retry.

Exit code: 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://recsys-workers-production.tanmainguyenbinh.workers.dev"
TRANSIENT_STATUSES = {502, 503, 504}
REQUEST_TIMEOUT = 120.0  # generous: covers ONNX lazy init on first semantic search
MAX_ATTEMPTS = 5
RETRY_SLEEP = 20.0

REQUIRED_METRICS = (
    "recsys_semantic_search_total",
    "recsys_keyword_fallback_total",
    "recsys_onnx_inference_seconds",
    "recsys_qdrant_requests_total",
    "recsys_qdrant_errors_total",
    "recsys_qdrant_latency_seconds",
)


class SmokeFailure(Exception):
    """Contract violation or exhausted retries."""


def _get_with_retry(client: httpx.Client, url: str) -> httpx.Response:
    """GET with controlled transient-only retry (no retry storm)."""
    last = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code < 400:
                return resp
            if resp.status_code in TRANSIENT_STATUSES:
                last = f"transient HTTP {resp.status_code}"
            else:  # 4xx / other: never retry
                raise SmokeFailure(f"{url} -> non-retryable HTTP {resp.status_code}")
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = type(e).__name__
        if attempt < MAX_ATTEMPTS:
            print(f"    attempt {attempt}/{MAX_ATTEMPTS} failed ({last}); retrying...")
            time.sleep(RETRY_SLEEP)
    raise SmokeFailure(f"{url} -> {last} after {MAX_ATTEMPTS} attempts")


def check_health(client: httpx.Client, base: str) -> str:
    body: dict[str, Any] = _get_with_retry(client, f"{base}/api/health").json()
    problems = []
    if body.get("model_loaded") is not True:
        problems.append(f"model_loaded={body.get('model_loaded')!r}")
    if body.get("qdrant_available") is not True:
        problems.append(f"qdrant_available={body.get('qdrant_available')!r}")
    if not body.get("users", 0) > 0 or not body.get("items", 0) > 0:
        problems.append(f"users/items={body.get('users')}/{body.get('items')}")
    if problems:
        raise SmokeFailure("health contract violated: " + "; ".join(problems))
    return f"users={body['users']} items={body['items']}"


def check_recommend(client: httpx.Client, base: str) -> str:
    body: dict[str, Any] = _get_with_retry(client, f"{base}/api/recommend/ml_user_429").json()
    items = body.get("items") or []
    if not items:
        raise SmokeFailure("recommend returned no items")
    bad = [i for i in items[:3] if not isinstance(i.get("score"), (int, float))]
    if bad:
        raise SmokeFailure(f"recommend items have non-numeric scores: {bad}")
    return f"{len(items)} items, top score={items[0]['score']}"


def check_search(client: httpx.Client, base: str) -> str:
    """THE critical gate: semantic search must actually be semantic."""
    body: dict[str, Any] = _get_with_retry(client, f"{base}/api/search?q=star").json()
    source = body.get("source")
    if source != "semantic":
        # Contract violation -> immediate fail (already retried by _get_with_retry
        # only for transport-level issues; a keyword response is definitive).
        raise SmokeFailure(f"expected source='semantic', got source={source!r}")
    if not body.get("items"):
        raise SmokeFailure("semantic search returned no items")
    return f"source=semantic, {len(body['items'])} items"


def check_metrics(client: httpx.Client, base: str) -> str:
    text = _get_with_retry(client, f"{base}/api/metrics").text
    missing = [m for m in REQUIRED_METRICS if m not in text]
    if missing:
        raise SmokeFailure(f"metrics missing: {missing}")
    return f"{len(REQUIRED_METRICS)}/{len(REQUIRED_METRICS)} P3 metrics exposed"


CHECKS = [
    ("health", check_health),
    ("recommend", check_recommend),
    ("semantic search", check_search),
    ("metrics", check_metrics),
]


def run(base_url: str) -> bool:
    total_start = time.monotonic()
    results: list[tuple[str, bool, str]] = []
    with httpx.Client(follow_redirects=True) as client:
        for i, (name, fn) in enumerate(CHECKS, 1):
            try:
                detail = fn(client, base_url.rstrip("/"))
                results.append((name, True, detail))
                print(f"[{i}/{len(CHECKS)}] {name:<22} PASS  ({detail})")
            except SmokeFailure as e:
                results.append((name, False, str(e)))
                print(f"[{i}/{len(CHECKS)}] {name:<22} FAIL")
                print(f"    Reason: {e}")
                break
    ok = all(r[1] for r in results) and len(results) == len(CHECKS)
    elapsed = time.monotonic() - total_start
    print()
    print(f"SMOKE TEST: {'PASS' if ok else 'FAIL'}  ({elapsed:.1f}s)")
    return ok


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    print(f"Smoke testing {base}")
    return 0 if run(base) else 1


if __name__ == "__main__":
    sys.exit(main())
