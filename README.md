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
scripts/              # real-time worker, one-off scripts
Dockerfile            # multi-stage production Dockerfile
docker-compose.yml    # local dev stack (api + qdrant + redis + redpanda)
.nginx              # nginx config (optional proxy)
Makefile              # automation targets (train, embeddings, test, worker, monitor)
Dockerfile            # production multi-stage
uv.lock               # frozen lockfile for reproducible installs
pyproject.toml        # project metadata + dependencies
```

## Requirements

- Python 3.12 (via `uv`), Docker / Docker Compose
- [uv](https://docs.astral.sh/uv/) as the package & environment manager
- Cloudflare account (Workers + Pages free tier)
- Render account (free tier for Docker deployment)
- Qdrant Cloud account (free 1GB cluster)

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
`MOVIELENS_VERSION`, ...). Secrets go in the Cloudflare Workers/`Render` env
vars — never commit `.env`.

## Deploy

### Local (Docker — chạy ngay, verify trước cloud)

```bash
# 1. Khởi tạo stack: API + Qdrant vector store
docker compose up --build -d

# 2. Train two-tower checkpoint (chọn: dùng model thật hoặc để demo fallback untrained)
make train   # tạo outputs/two_tower.pt (3 epochs trên CPU, mất ~5-10 phút)

# 3. Upload embeddings vào Qdrant (để enable semantic search)
#    -- Nên chạy một lần sau khi model train xong hoặc có dữ liệu items đầy đủ
uv run python -m training.embeddings --url http://localhost:6333 --collection items
#    -> In ra: 62423 items, dim=384, indexed=62423 in 'items'

# 4. Kiểm tra API hoạt động
curl http://localhost:8000/health          # {"status":"ok",...}
curl -s -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d '{"user_id":"ml_user_2262","top_n":3}'
curl -s -X POST http://localhost:8000/event -H "Content-Type: application/json" -d '{"user_id":"ml_user_2262","item_id":"ml_item_1","event_type":"click"}'
curl -s "http://localhost:8000/search?q=toy+story"   # source: semantic hoặc keyword fallback
curl -s http://localhost:8000/metrics            # Prometheus metrics text
```

### Cloud free-tier (Render + Qdrant Cloud + Cloudflare Workers)

#### 1. Render (free tier — Deploy API)
   - New Web Service → chọn Dockerfile (từ repo)
   - Environment variables (bắt buộc):
     - `QDRANT_URL=https://<your-qdrant-cloud-url>` (VD: `https://e891f845-c76f-4e54-bf4f-5bc649e459b5.us-west-1-0.aws.cloud.qdrant.io`)
     - `QDRANT_API_KEY=batch1` (JWT sub claim 'project')
     - `QDRANT_COLLECTION=items`
     - `MODEL_URL=https://raw.githubusercontent.com/imtarget05/Recommendation-System/main/outputs/two_tower.pt` (hoặc lưu checkpoint riêng ở Release)
     - `LLM_PROVIDER=stub` (mặc định, dùng stub LLM miễn phí, không cần Ollama)
     - `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
     - `REDIS_URL` / `KAFKA_BROKER_URL` (để khi đã có real-time pipeline, để API biết)

   - Build command: `uv sync --all-extras --dev` (hoặc留空 nếu dùng image pre-built)
   - Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

#### 2. Qdrant Cloud (free 1GB)
   - Tạo cluster trên qdrant.io (free tier)
   - Lấy `QDRANT_URL` và `QDRANT_API_KEY` ở trên
   - Chạy `make embeddings --url <QDRANT_URL> --collection items --api-key <API_KEY>` để upsert vector
   - Hoặc dùng API/Qdrant Client để upload qua dashboard

#### 3. Cloudflare Workers (wrangler — serve static + proxy API)
   - `wrangler init recsys-workers` (đã tạo sẵn trong repo)
   - Cấu hình `wrangler.jsonc`:
     ```toml
     [env.production]
     QDRANT_URL = "https://e891f845-c76f-4e54-bf4f-5bc649e459b5.us-west-1-0.aws.cloud.qdrant.io"
     QDRANT_API_KEY = "batch1"
     QDRANT_COLLECTION = "items"
     LLM_PROVIDER = "stub"
     EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
     
     # Route /api/* đến Render API service qua fetch
     API_RENDER_URL = "https://<render-service>.onrender.com"
     ```
   
   - Code `src/worker.ts`: import fetch, call `https://<render-service>.onrender.com/api/...`
   - Hoặc đơn giản hơn: Workers chỉ serve static `static/` và proxy `/api/` calls đến Render
   
   - Deploy: `wrangler publish --env production`

#### 5. Cloudflare Pages (serve UI tĩnh)
   - New Site → Import từ GitHub → chọn thư mục `static/`
   - Build command: `echo "done"`
   - Root directory: `/`

#### 6. Kiểm tra endpoints sau deploy
   ```bash
   # Health + model
   curl https://<render-service>/health
   
   # Recommend (dùng model thật nếu upload checkpoint, ngược lại vẫn serve untrained + scores)
   curl -s -X POST https://<render-service>/recommend -H "Content-Type: application/json" -d '{"user_id":"ml_user_2262","top_n":3}'
   
   # Event logging
   curl -s -X POST https://<render-service>/event -H "Content-Type: application/json" -d '{"user_id":"ml_user_2262","item_id":"ml_item_1","event_type":"click"}'
   
   # Search (semantic Qdrant nếu có vector, ngược lại keyword)
   curl "https://<render-service>/search?q=toy+story"
   
   # Prometheus metrics
   curl https://<render-service>/metrics | grep -E "^recsys_(http|events|rec)"
   
   # UI: mở https://<pages-service>.pages.dev
   ```

### Flow dữ liệu (local + cloud tương đồng)

```
User UI
   │
   ├─► GET /search?q=...     → Qdrant semantic (nếu online) hoặc keyword fallback
   │
   ├─► POST /recommend       → Two-Tower retrieval (model thật hoặc fallback untrained)
   │
   ├─► POST /event (click)   → API log event → buffer (in-memory, hoặc Redis/Kafka sau này)
   │
   └─► AI chat panel         → POST /chat/recommend → LLM stub → render items → click→event loop
```

### Retrain model với dữ liệu mới

```bash
# Retrain từ đầu (chạy trên CPU 3 epochs, mất ~5-10 phút)
make train

# Upload embeddings lại (nếu thêm item mới hoặc thay đổi dữ liệu)
uv run python -m training.embeddings --url http://localhost:6333 --collection items

# Deploy lại Render/Qdrant với file checkpoint mới
# - Upload outputs/two_tower.pt lên GitHub Release → set MODEL_URL
# - Hoặc mount volume qua DATA_BASE_URL nếu dùng data cục bộ
```

### Lưu ý quan trọng (data separation giữa projects)

- Mỗi project nên có Qdrant cluster/tên collection riêng (VD: `recsys-batch1`, `recsys-batch2`) để tránh chồng chéo dữ liệu vector
- Environment variables QDRANT_URL/QDRANT_API_KEY/QDRANT_COLLECTION nên rõ ràng gắn với project đang deploy
- Render service env variables nên đặt tên gợi ý (VD: `RECsys_QDRANT_URL` chứ không chung chung)
- Checkpoint model (two_tower.pt) nên upload riêng theo release hoặc dùng `MODEL_URL` trỏ thẳng về file, tránh confusion với data artifacts khác
```

Now let me also verify the worker exists and the full repo state:
<tool_call>
<function=bash>
<parameter=workdir>
/Users/mainguyenbinhtan/Downloads/Recommendation System/realtime-recsys