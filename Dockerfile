# Real-time recommendation system — production server (multi-stage, uv frozen)
# Build:  docker compose up --build
# Cloud:  Render free tier web service from this Dockerfile (stateless: model + data
#         pulled from MODEL_URL / DATA_BASE_URL at startup, see api/main.py).

# ── Stage 1: builder (resolve + compile all deps from uv.lock) ──
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS builder

WORKDIR /app

# Lockfile first for layer caching
COPY pyproject.toml uv.lock ./
COPY app ./app
COPY config ./config

# Frozen lockfile install: wheels pre-built in builder; runtime stays slim.
# (Dev extras pytest/ruff are intentionally NOT shipped into the served image to
#  keep the build + image small enough for Render's free starter buildPlan.)
RUN uv sync --frozen --python 3.12

# ── Stage 2: runtime (slim, no build tools) ──
FROM python:3.12-slim-trixie

WORKDIR /app

# Runtime system deps only (gcc/g++ NOT needed — wheels are already built)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# App code (exclude data/ outputs/ via .dockerignore)
COPY . .

# NOTE: the embedding model (sentence-transformers/all-MiniLM-L6-v2) is now loaded
# lazily by api/main.py (_get_embedder) on first /search and cached afterward,
# so it is NOT pre-downloaded at build time (keeps the image slim / fits free tier).

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Stateless serving: DATA_BASE_URL/MODEL_URL pull artifacts at startup on cloud
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]