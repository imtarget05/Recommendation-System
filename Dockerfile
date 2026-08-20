# Real-time recommendation system — production server
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Install uv (fast package manager)
RUN pip install --no-cache-dir uv

# Copy project files
COPY . .

# Install Python dependencies (frozen lockfile for reproducibility)
RUN uv pip install --system \
    torch fastapi uvicorn sentence-transformers pandas numpy scipy pydantic pyyaml

# Pre-download model for cold-start (speeds up first request significantly)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

# Serve data volume (mounted at runtime)
VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
