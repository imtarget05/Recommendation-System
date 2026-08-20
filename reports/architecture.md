# Architecture & Phase Progress

## Phase 0 — Repository inspection (2026-08-19)

Initial state: only `realtime-recsys-ai-master-spec.md` present (greenfield).

Environment: Python 3.14.6 system / uv 0.12.0 / Docker 29.6.1 + Compose v5.2.0 /
8 CPUs / 16 GB RAM.

Decisions:
- **Python 3.12 pinned via uv** (system 3.14 too new for PyTorch/LightGBM wheels).
- Single repo `realtime-recsys/`; internal schema defined once in `app/schemas`.
- Data layer built dataset-agnostic (adapter registry) per spec Section 5.3.

## Phase 1 — Data (DONE, verified, finalized 2026-08-20)

### Implemented
- Internal schema: `interactions (user_id, item_id, event_type, timestamp)`,
  `items (item_id, title, category, tags, text_description)`,
  `users (user_id [, metadata])` — `metadata` is an optional JSON-string column
  (Section 5.4).
- Adapters: `movielens` (ml-25m + ml-latest-small, auto-download from grouplens),
  `hm` (Kaggle H&M, requires KAGGLE_USERNAME/KAGGLE_KEY + `kagglehub`),
  `huggingface` (generic column-hint mapping).
- Normalization: implicit events `view/like` (MovieLens rating >= 4 -> like),
  config-driven weights `view=1, click=2, like=3, purchase=5`, recency decay
  `exp(-lambda * age)` with configurable `recency.unit` (days/hours/minutes/seconds)
  and `recency.max_age_days` floor, item text builder.
- Time-based split (`global` chronological 70/10/20 + `per-user` mode), deterministic.
- Dataset manifest auto-updated at ingest: source, version, row_count, checksum,
  schema fields, split sizes (`data/manifests/dataset_manifest.yaml`).
- Max-row cap: default `strided` mode keeps the full temporal span (min + max
  timestamps retained); `head` mode available for tiny quick subsets.

### Finalization pass (bug fixes / spec alignment)
- `validate_users` previously dropped the H&M `metadata` column; it now keeps
  optional columns (`USER_OPTIONAL_COLUMNS`) and the H&M loader stores metadata
  as a JSON string (parquet-safe). Verified by a synthetic H&M loader test.
- `recency_weight` honors `recency.unit` and `recency.max_age_days` from config
  (previously hard-coded days / no cap).
- `INTERACTION_DTYPES["timestamp"]` corrected to a tz-aware datetime dtype.
- `trim_to_max_rows` default becomes strided sampling (no longer drops the most
  recent interactions when capping `max_rows`).
- H&M loader emits canonical interaction column order; per-user split respects
  the configured `min_interactions`.
- HM adapter covered by an offline unit test using synthetic Kaggle-layout CSVs
  (fill empty `detail_desc` without fabricating text).

### Verification evidence (final state, canonical = ml-latest-small)
```
$ uv run pytest                          -> green (unit suite; count grows as later
                                           phases add tests), ruff clean

$ uv run python -m data.scripts.ingest --dataset movielens --version ml-latest-small --max-rows 100000
manifest: movielens ml-latest-small rows=100000
splits  {train: 70,000, validation: 10,000, test: 20,000}
train/validation/test non-overlapping (chronological), UTC timestamps, parquet OK
events: view=78,416  like=21,584

$ uv run python -m data.scripts.ingest --dataset movielens --version ml-25m   (5M-row cap; regen: make ingest-25m)
train      rows= 3,500,000  users= 24,558  span=1995-01-09 -> 2015-03-23
validation rows=   500,000  users=  3,886  span=2015-03-23 -> 2016-07-19
test       rows= 1,000,000  users=  6,144  span=2016-07-19 -> 2019-11-21
events: view=3,834,894  like=1,165,106   (numbers from the pre-finalization run:
the exact veto row mix changes slightly with strided trimming)
```

Sample items (ml-latest-small) confirmed: title/category/tags/text_description populated.

### Known limitations / notes
- H&M adapter requires Kaggle credentials; ingest not run here (documented in README),
  but normalization + metadata handling is covered by the unit test.
- `max_rows` cap defaults to 5,000,000 (config `data.max_rows`).
- HM dataset only contains `purchase` events (no views/clicks) — honest mapping,
  no fabricated events.
- Cross-dataset id prefixes (`ml_user_*`, `hm_item_*`) prevent collisions.
- The manifest reflects the last ingest run (currently `ml-latest-small`); run
  `make ingest-25m` when the 25m processed artifact is needed again.

### Definition of done
- [x] processed dataset exists (parquet for train/validation/test + items + users)
- [x] schema validated (unit tests + runtime validation)
- [x] sample inspected (printed in ingest output)
- [x] split generated (time-based, deterministic, non-overlapping)
- [x] optional user metadata preserved end-to-end (schema §5.4) + recency config honored (§6.3)

## Next phases (roadmap)
2. Baselines (Popularity, CF) — metrics on time-based split.
3. Embeddings (item text -> MiniLM-L6-v2 -> Qdrant).
4. Two-Tower retrieval.
5. LightGBM ranking.
6. FastAPI serving.
7. Real-time pipeline (Redpanda/Kafka producer+consumer, Redis state).
8. LLM layer (prompt registry, structured output, fallback).
9. Monitoring & dashboards.
10. Docker/CI/CD.
11. Cloud deployment.
12. Kubernetes manifests (optional).
13. Final documentation & reports.

## Phase 2 — Baselines (DONE, verified)

Popularity (weighted counts, k=20) and CF-ALS (factors=64, iters=40, alpha=5) were trained and evaluated on the time-based split (70/10/20). The full experiment grid (alpha ∈ {5,20,40} × iters ∈ {15,40}) and results are recorded in . The final configuration is cf_als v3 (iterations=40, alpha=5) and popularity v1.

## Phase 2 — Data

The canonical dataset adapter chain ( /  / ) produces the internal normalized schema (). The deterministic time-strided cap preserves temporal spread and item catalog coverage (, default mode=). Global split at 70/10/20 with per-user ground truth (excludes user-specific train items, restricts to model catalog). Dataset manifest auto-generated at .

### Verification checklist (DONE)

- [x] processed dataset exists (parquet for train/validation/test + items + users)
- [x] schema validated (unit tests + runtime validation)
- [x] sample inspected (printed in ingest output)
- [x] split generated (time-based, deterministic)

