#!/usr/bin/env python3
"""Poll Render deployment status for a given commit until it reaches a final state.

Usage: wait_render_deploy.py <service_id> <commit_sha> [timeout_seconds]
Requires RENDER_API_KEY env var. Exit 0 on live, 1 on failure/timeout.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

TERMINAL_FAIL = {"build_failed", "update_failed", "canceled", "pre_deploy_failed", "deactivated"}


def main() -> int:
    svc = sys.argv[1]
    sha = sys.argv[2][:7]
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 900
    key = os.environ.get("RENDER_API_KEY", "")
    if not key:
        print("RENDER_API_KEY not set")
        return 1

    url = f"https://api.render.com/v1/services/{svc}/deploys?limit=10"
    deadline = time.time() + timeout
    last = "?"
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
            if not isinstance(data, list):
                print(f"[poll] unexpected API payload: {str(data)[:200]}")
                print("auth_or_api_error")
                return 1
            for entry in data:
                d = entry.get("deploy", {})
                commit = d.get("commit") or {}
                cid = str(commit.get("id", "") if isinstance(commit, dict) else commit)
                if cid.startswith(sha):
                    status = d.get("status", "?")
                    # API may return status as a plain string or {"name": "..."}
                    last = (
                        status.get("name", "?")
                        if isinstance(status, dict)
                        else str(status)
                    )
                    break
            else:
                last = "not_found"
        except Exception as e:  # noqa: BLE001
            print(f"[poll] request error: {type(e).__name__}")
            last = "request_error"
        print(f"[poll] {sha} -> {last}", flush=True)
        if last == "live":
            return 0
        if last in TERMINAL_FAIL:
            return 1
        time.sleep(15)
    print("timeout waiting for live deployment")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
