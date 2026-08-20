#!/usr/bin/env bash
# Run the recommendation API locally (development mode)
# Usage: ./run.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "Starting RecSys API..."
echo "  API: http://localhost:8000"
echo "  Docs: http://localhost:8000/docs"

exec uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
