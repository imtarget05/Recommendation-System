# Real-time recommendation system — production server (multi-stage, uv frozen)
# Free-tier serving: NumPy retriever + ONNX MiniLM — NO torch in the runtime image.
# Build:  docker compose up --build
# Cloud:  Render free tier web service from this Dockerfile (stateless: embeddings.npz
#         + data pulled from EMBEDDINGS_URL / DATA_BASE_URL / SEMANTIC_ASSETS_URL at startup).

# ── Stage 1: builder (resolve + compile PRODUCTION deps only) ──
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS builder

WORKDIR /app

# Lockfile first for layer caching
COPY pyproject.toml uv.lock ./
COPY app ./app
COPY config ./config

# Production dependencies ONLY: export base deps from uv.lock (no extras —
# torch / sentence-transformers live in [training], pytest/ruff in [dev]) and
# pip-install them into a clean venv. (uv sync would install ALL extras.)
RUN uv export --frozen --no-dev --no-default-groups --no-emit-project \
      --format requirements-txt -o /tmp/requirements.txt \
    && uv venv /app/.venv --python 3.12 \
    && uv pip install --python /app/.venv/bin/python -r /tmp/requirements.txt

# ── Stage 2: runtime (slim, no build tools) ──
FROM python:3.12-slim-trixie

WORKDIR /app

# Runtime system deps only (gcc/g++ NOT needed — wheels are already built)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# App code (exclude data/ outputs/ artifacts/ via .dockerignore)
COPY . .

# Semantic search assets (ONNX MiniLM + tokenizer) are downloaded lazily from
# SEMANTIC_ASSETS_URL on first /search (see api/main.py _get_embedder) — keeps
# the image slim and the startup memory low for the free tier.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Stateless serving: DATA_BASE_URL/EMBEDDINGS_URL pull artifacts at startup on cloud
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]