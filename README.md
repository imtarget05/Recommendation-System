# RealTime RecSys — Personalized Recommendation & AI Recommendation Platform

Real-time, end-to-end recommendation system built as an interview-grade portfolio
project. It covers dataset adapters, offline training (popularity / CF / two-tower /
LightGBM ranking), vector retrieval (Qdrant), real-time event pipeline
(Redpanda/Kafka + Redis), an LLM conversational layer (Qwen3/Gemma) that never
replaces the recommender engine, monitoring (Prometheus/Grafana) and CI/CD.

> Status: implemented in phases. See `reports/architecture.md` for the roadmap and
> `reports/experiments.md` for measured results. Every phase has tests and a runnable
> artifact; nothing is assumed working without inspection.

## Project layout (live view)

```text
app/                  # shared code: config, internal schemas (used by serving too)
config/config.yaml    # central non-secret configuration
data/
  connectors/         # dataset adapters -> internal schema (movielens, hm, huggingface)
  preprocessing/      # validation, weighting, recency, time-based split
  scripts/            # ingest CLI + dataset manifest
  manifests/          # dataset_manifest.yaml (versioning metadata)
tests/                # unit tests (integration/api added in later phases)
```

## Requirements

- Python 3.12 (via `uv`), Docker / Docker Compose
- [uv](https://docs.astral.sh/uv/) as the package & environment manager

## Quick start (Phase 1 — data)

```bash
uv sync --all-extras --dev   # install deps into .venv (Python 3.12)
make test                    # run unit tests
make ingest-small            # MovieLens ml-latest-small (fast, pipeline validation)
make ingest-25m              # MovieLens ml-25m (default 5M-row cap, see config)
```

Roll your own dataset:

```bash
uv run python -m data.scripts.ingest --dataset movielens --version ml-latest-small
uv run python -m data.scripts.ingest --dataset hm                 # needs KAGGLE_USERNAME/KAGGLE_KEY
uv run python -m data.scripts.ingest --dataset huggingface --version <repo_id>
```

Outputs (all gitignored):

```text
data/processed/interactions_{train,validation,test}.parquet
data/processed/items.parquet          # internal item schema
data/processed/users.parquet          # internal user schema
data/manifests/dataset_manifest.yaml  # versioning metadata (source, checksum, splits)
```

## Configuration

Non-secret defaults live in `config/config.yaml` (weights, recency lambda, split
ratios). Environment variables override key fields (`DATA_DIR`, `DATASET_NAME`,
`MOVIELENS_VERSION`, ...). Secrets go in `.env` (see `.env.example`); never commit
`.env`.

## Verification convention

Every phase is verified by: unit tests -> real run -> inspect output -> record
evidence in `reports/`. Current Phase 1 evidence: see the "Phase 1 — Data" section
in `reports/architecture.md`.

## License

MIT (see `LICENSE`).