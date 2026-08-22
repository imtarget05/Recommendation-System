#!/usr/bin/env python3
"""Controlled load test for the free-tier recsys ($0/month).

Measures baseline latency, concurrency breakpoints, and semantic-contract
correctness. NON-destructive: only hits GET /api/health, /api/recommend,
and /api/search?q=. Safe to run against production.

Usage:
    python3 scripts/load_test.py --url <WORKER_URL> \
        --endpoint {health,recommend,search,mixed} \
        --concurrency 10 --requests 50 --timeout 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

USER = "ml_user_429"
QUERY = "star"


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    idx = max(0, min(len(s) - 1, round(len(s) * q / 100)))
    return s[idx]


async def hit(
    client: httpx.AsyncClient, base: str, ep: str
) -> tuple[bool, float, int, dict | None, str | None]:
    """Returns (ok, latency_ms, status, body, error_or_contract_violation)."""
    url = f"{base}/api/search?q={QUERY}"
    if ep == "recommend":
        url = f"{base}/api/recommend/{USER}"
    elif ep == "health":
        url = f"{base}/api/health"
    t0 = time.perf_counter()
    try:
        r = await asyncio.wait_for(client.get(url), timeout=25)
    except (TimeoutError, httpx.HTTPError) as e:
        return False, (time.perf_counter() - t0) * 1000, 0, None, f"transport:{type(e).__name__}"
    dt = (time.perf_counter() - t0) * 1000
    body: dict | None = None
    err: str | None = None
    try:
        body = r.json()
    except Exception:
        err = "invalid_json"
    ok = (r.status_code == 200) and (body is not None)
    # Contract check
    if ok and ep == "search":
        if body.get("source") != "semantic":
            ok = False
            err = f"source:{body.get('source')}"
    if ok and ep == "health":
        if not body.get("model_loaded") or not body.get("qdrant_available"):
            ok = False
            err = f"health:{body}"
    return ok, dt, r.status_code, body, err


async def run_level(base: str, ep: str, concurrency: int, total: int, timeout: int) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def one() -> tuple[bool, float, int, dict | None, str | None]:
        async with sem:
            return await hit(client, base, ep)

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(max_connections=concurrency,
                            max_keepalive_connections=concurrency),
    ) as client:
        t_start = time.perf_counter()
        tasks = [asyncio.create_task(one()) for _ in range(total)]
        # mixed: interleave search + recommend
        if ep == "mixed":
            for i, t in enumerate(tasks):
                ep_choice = "search" if i % 5 in (0, 1, 2) else "recommend"
                t._ep = ep_choice  # stash
        res = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t_start

    lat = [r[1] for r in res]
    ok_count = sum(1 for r in res if r[0])
    err_types: dict[str, int] = {}
    for r in res:
        if not r[0]:
            k = r[3] or r[4] or f"status_{r[2]}"
            err_types[k] = err_types.get(k, 0) + 1
    return {
        "endpoint": ep,
        "concurrency": concurrency,
        "requests": total,
        "success": ok_count,
        "success_rate": round(ok_count / total, 4),
        "errors": err_types,
        "p50_ms": round(pct(lat, 50), 1),
        "p95_ms": round(pct(lat, 95), 1),
        "p99_ms": round(pct(lat, 99), 1),
        "min_ms": round(min(lat), 1),
        "max_ms": round(max(lat), 1),
        "mean_ms": round(statistics.mean(lat), 1),
        "median_ms": round(statistics.median(lat), 1),
        "total_elapsed_s": round(elapsed, 2),
        "throughput_rps": round(total / elapsed, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--endpoint",
                    choices=["health", "recommend", "search", "mixed"],
                    default="mixed")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # Mixed endpoint: run separate search/recommend tasks
    ep_use = args.endpoint
    base = args.url.rstrip("/")
    print(f"[load] {ep_use} x{args.requests} @ concurrency {args.concurrency} -> {base} ...")
    result = asyncio.run(run_level(base, ep_use, args.concurrency, args.requests, args.timeout))
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0 if result["success_rate"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
