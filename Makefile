.PHONY: install lint test train embeddings worker monitor-drift ingest-25m ingest-small ingest-hm clean

install:
	uv sync --all-extras

lint:
	uv run ruff check .

test:
	uv run pytest

train:
	uv run python training/train_two_tower.py

embeddings:
	uv run python -m training.embeddings

worker:
	uv run python scripts/real_time_worker.py

monitor-drift:
	@echo "--- RecSys Drift Detection ---"
	uv run python -c "
from prometheus_client import Counter, Histogram
# Read current metric values via HTTP (simulated)
# In practice: curl http://localhost:8000/metrics | grep recsys_
print('Drift check: compare current prometheus counters against baseline')
print('Use: make monitor-dashboard to open Grafana UI')
"

monitor-dashboard:
	@echo "--- RecSys Grafana Dashboard ---"
	@echo "Dashboard JSON: reports/grafana_dashboard.json"
	@echo "To load: start Grafana and import this JSON, or use:"

ingest-small:
	uv run python -m data.scripts.ingest --dataset movielens --version ml-latest-small --max-rows 100000

ingest-25m:
	uv run python -m data.scripts.ingest --dataset movielens --version ml-25m

ingest-hm:
	uv run python -m data.scripts.ingest --dataset hm

clean:
	rm -rf data/raw data/processed data/features data/embeddings .pytest_cache .ruff_cache