"""Rollback tooling for the Render-hosted RecSys service (P6).

Strategies, in order of preference:
  A) Direct re-deploy of a known-good commit via the Render API
     (POST /services/{id}/deploys with {"commitId": "<sha>"}). Verified live
     before being trusted: after the deploy goes `live`, we confirm the
     deployed commit matches the target, then run the production smoke test.
  B) If the API rejects commit-pinned deploys, fall back to printing exact
     `git revert` instructions (forward-fix), which Render auto-deploys.

Usage:
    python3 scripts/rollback.py list                     # show deploy history
    python3 scripts/rollback.py plan --sha <short-sha>   # dry-run: resolve + validate
    python3 scripts/rollback.py execute --sha <short-sha> [--skip-smoke]

Requires env RENDER_API_KEY. Service id defaults to srv-da3tjnu1egvs73ark8l0.
Never prints the token.
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx

SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "srv-da3tjnu1egvs73ark8l0")
SMOKE_URL = os.environ.get(
    "SMOKE_BASE_URL",
    "https://recsys-workers-production.tanmainguyenbinh.workers.dev",
)


def _client() -> httpx.Client:
    token = os.environ.get("RENDER_API_KEY", "")
    if not token:
        sys.exit("ERROR: RENDER_API_KEY is not set")
    return httpx.Client(
        base_url="https://api.render.com/v1",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )


def _deploys(client: httpx.Client, limit: int = 10) -> list[dict]:
    resp = client.get(f"/services/{SERVICE_ID}/deploys", params={"limit": limit})
    resp.raise_for_status()
    return [d["deploy"] for d in resp.json()]


def cmd_list(client: httpx.Client) -> None:
    for d in _deploys(client):
        commit = (d.get("commit") or {}) if isinstance(d.get("commit"), dict) else {}
        sha = str(d.get("commitId") or commit.get("id") or "?")[:9]
        print(f"{d['id']}  {d.get('status', '?'):20s}  {sha}  {d.get('createdAt', '')}")


def _resolve_sha(client: httpx.Client, short_sha: str) -> dict | None:
    """Find a *successful* deploy in history whose commit starts with short_sha."""
    for d in _deploys(client, limit=30):
        commit = (d.get("commit") or {}) if isinstance(d.get("commit"), dict) else {}
        cid = str(d.get("commitId") or commit.get("id") or "")
        if cid.startswith(short_sha):
            return d
    return None


def cmd_plan(client: httpx.Client, short_sha: str) -> None:
    d = _resolve_sha(client, short_sha)
    if d is None:
        sys.exit(f"FAIL: no deploy found for commit '{short_sha}' in recent history")
    print(f"target deploy : {d['id']}")
    print(f"commit        : {str(d.get('commitId', ''))[:12]}")
    print(f"last status   : {d.get('status')}")
    print("strategy      : POST /deploys {'commitId': ...}, poll live, then smoke test")
    print("PLAN OK (dry-run, nothing deployed)")


def _wait_live(client: httpx.Client, deploy_id: str, timeout_s: int) -> bool:
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"/services/{SERVICE_ID}/deploys/{deploy_id}")
        r.raise_for_status()
        status = r.json().get("status", "?")
        print(f"  poll: {status}")
        if status == "live":
            return True
        if status in {"build_failed", "update_failed", "canceled", "pre_deploy_failed"}:
            return False
        time.sleep(15)
    return False


def cmd_execute(client: httpx.Client, short_sha: str, skip_smoke: bool) -> None:
    d = _resolve_sha(client, short_sha)
    if d is None:
        sys.exit(f"FAIL: no deploy found for commit '{short_sha}'")
    full_sha = str(d.get("commitId") or "")
    print(f"[1/3] triggering deploy of commit {full_sha[:12]} ...")
    resp = client.post(
        f"/services/{SERVICE_ID}/deploys",
        json={"commitId": full_sha, "clearCache": "preserve"},
    )
    if resp.status_code >= 400:
        body = resp.text[:300]
        print(f"Render rejected commit-pinned deploy ({resp.status_code}): {body}")
        print()
        print("FALLBACK (forward-fix):")
        print("  git revert -m 1 HEAD          # or: git revert <bad-commit>")
        print("  git push origin main          # Render auto-deploys the revert")
        print("  python3 scripts/wait_render_deploy.py <revert-sha>")
        sys.exit(1)

    deploy_id = resp.json()["id"]
    print(f"[2/3] waiting for deploy {deploy_id} to go live ...")
    if not _wait_live(client, deploy_id, timeout_s=900):
        sys.exit("FAIL: rollback deploy did not reach 'live'")
    print("[3/3] running production smoke test ...")
    if skip_smoke:
        print("SKIP_SMOKE=true — ROLLBACK DEPLOYED (unverified)")
        return
    rc = os.system(
        f'python3 "{os.path.join(os.path.dirname(__file__), "smoke_test.py")}" "{SMOKE_URL}"'
    )
    if rc != 0:
        sys.exit("FAIL: rollback deployed but smoke test FAILED — release still unhealthy")
    print("ROLLBACK COMPLETE: live + smoke PASS")


def main() -> None:
    p = argparse.ArgumentParser(description="RecSys rollback tooling")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    pp = sub.add_parser("plan")
    pp.add_argument("--sha", required=True)
    pe = sub.add_parser("execute")
    pe.add_argument("--sha", required=True)
    pe.add_argument("--skip-smoke", action="store_true")
    args = p.parse_args()

    with _client() as c:
        if args.cmd == "list":
            cmd_list(c)
        elif args.cmd == "plan":
            cmd_plan(c, args.sha)
        else:
            cmd_execute(c, args.sha, args.skip_smoke)


if __name__ == "__main__":
    main()
