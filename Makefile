.PHONY: install lint test train embeddings ingest-25m ingest-small ingest-hm clean

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

ingest-small:
	uv run python -m data.scripts.ingest --dataset movielens --version ml-latest-small --max-rows 100000

ingest-25m:
	uv run python -m data.scripts.ingest --dataset movielens --version ml-25m

ingest-hm:
	uv run python -m data.scripts.ingest --dataset hm

clean:
	rm -rf data/raw data/processed data/features data/embeddings .pytest_cache .ruff_cache