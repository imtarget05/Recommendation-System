#!/usr/bin/env bash
# Production smoke test (P5) — run after every deployment.
# Verifies content, not just HTTP 200:
#   /health   → model_loaded:true, qdrant_available:true
#   /recommend→ non-empty items with scores
#   /search   → source:"semantic"
# Usage: ./scripts/smoke_test.sh [BASE_URL]
#        BASE_URL defaults to the Cloudflare Worker; pass the Render URL to test direct.
set -euo pipefail

BASE="${1:-https://recsys-workers-production.tanmainguyenbinh.workers.dev}"
# Worker proxies /api/* → Render; direct Render URL needs no prefix mapping.
if [[ "$BASE" == *"workers.dev"* ]]; then
  H="$BASE/api/health"; R="$BASE/api/recommend/ml_user_429"; S="$BASE/api/search?q=star"
else
  H="$BASE/health"; R="$BASE/recommend/ml_user_429"; S="$BASE/search?q=star"
fi

fail() { echo "❌ SMOKE FAIL: $1"; exit 1; }
jget() { python3 -c "import sys,json; d=json.load(sys.stdin); print(eval(\"d$1\"))" 2>/dev/null; }

echo "── [1/3] GET $H"
HEALTH=$(curl -sf -m 90 "$H") || fail "/health unreachable"
echo "$HEALTH" | jget "['model_loaded']" | grep -q True || fail "model_loaded != true: $HEALTH"
echo "$HEALTH" | jget "['qdrant_available']" | grep -q True || fail "qdrant_available != true: $HEALTH"

echo "── [2/3] GET $R"
REC=$(curl -sf -m 60 "$R") || fail "/recommend unreachable"
N=$(echo "$REC" | jget "['items']" | python3 -c "import ast,sys; print(len(ast.literal_eval(sys.stdin.read())))")
[ "${N:-0}" -gt 0 ] || fail "/recommend returned no items: $(echo "$REC" | head -c 200)"

echo "── [3/3] GET $S"
SEARCH=$(curl -sf -m 120 "$S") || fail "/search unreachable"
SRC=$(echo "$SEARCH" | jget "['source']")
[ "$SRC" = "semantic" ] || fail "/search source='$SRC' (expected semantic): $(echo "$SEARCH" | head -c 200)"

echo "✅ SMOKE PASS — health ok, recommend returned $N items, search source=semantic"
