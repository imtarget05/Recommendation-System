"""Real-time serving API for Two-Tower recommendation system.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                     — frontend HTML page
    GET  /health               — liveness probe
    GET  /metrics              — Prometheus metrics
    GET  /recommend/{user_id}  — top-K recommendations for a known user
    POST /recommend            — batch recommend (Section 14)
    POST /event                — log a user interaction event (Section 14)
    POST /chat/recommend       — conversational recommendation
    GET  /popular              — popular items (cold-start fallback)
    GET  /search?q=...         — keyword search over item titles

Stateless deployment (Section 23): if the trained checkpoint and processed data
files are absent, they are downloaded from MODEL_URL / DATA_BASE_URL at startup.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from app.numpy_retriever import NumpyRetriever, load_retriever
from app.query_norm import normalize_query, strip_accents
from app.rerank import content_tokens, score_candidate
from llm.service import LLMService
from training.common import build_weighted_counts

DATA_DIR = os.environ.get("DATA_DIR", "data/processed")
MODEL_PATH = os.environ.get("EMBEDDINGS_PATH", "artifacts/embeddings.npz")
MODEL_URL = os.environ.get("EMBEDDINGS_URL", "")  # npz artifact (converted from two_tower.pt)
DATA_BASE_URL = os.environ.get("DATA_BASE_URL", "")
SEMANTIC_ASSETS_URL = os.environ.get("SEMANTIC_ASSETS_URL", "")  # hosts model.onnx + tokenizer.json
MODEL_VERSION = os.environ.get("MODEL_VERSION", "two-tower-v1")
MANIFEST_URL = os.environ.get("MANIFEST_URL", "")  # versioned artifact manifest (P2)
QDRANT_URL = os.environ.get("QDRANT_URL", "https://e891f845-c76f-4e54-bf4f-5bc649e459b5.us-west-1-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "batch1")  # JWT sub claim 'project' = batch1
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "items")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

EVENT_TYPES = {"view", "click", "like", "purchase", "rate"}

# ═══════════════════════════════════════════════════════════════
# Prometheus metrics
# ═══════════════════════════════════════════════════════════════

HTTP_REQUESTS = Counter("recsys_http_requests_total", "HTTP requests", ["method", "path"])
HTTP_LATENCY = Histogram(
    "recsys_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
REC_REQUESTS = Counter("recsys_rec_requests_total", "Recommendation requests", ["endpoint"])
EVENTS = Counter("recsys_events_total", "User interaction events", ["event_type"])
SEMANTIC_SEARCH_TOTAL = Counter(
    "recsys_semantic_search_total", "Searches served semantically via Qdrant"
)
KEYWORD_FALLBACK_TOTAL = Counter(
    "recsys_keyword_fallback_total", "Searches that fell back to keyword matching"
)
ONNX_INFERENCE_SECONDS = Histogram(
    "recsys_onnx_inference_seconds", "ONNX encoder inference latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
QDRANT_REQUESTS = Counter("recsys_qdrant_requests_total", "Qdrant queries issued")
QDRANT_ERRORS = Counter(
    "recsys_qdrant_errors_total", "Qdrant query failures", ["error_type"]
)
QDRANT_LATENCY = Histogram(
    "recsys_qdrant_latency_seconds", "Qdrant query latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

_ERROR_TYPES = ("timeout", "connection", "unavailable", "other")


def _classify_qdrant_error(exc: Exception) -> str:
    """Map an exception to a finite error_type enum (no free-form labels)."""
    s = f"{type(exc).__name__}: {exc}".lower()
    if "timeout" in s or "timed out" in s:
        return "timeout"
    if "connect" in s or "network" in s or "dns" in s:
        return "connection"
    if "401" in s or "403" in s or "unauthorized" in s or "forbidden" in s:
        return "unavailable"
    return "other"


class RecItem(BaseModel):
    item_id: str
    title: str
    category: str
    score: float


class RecommendResponse(BaseModel):
    user_id: str
    items: list[RecItem]
    latency_ms: float
    model: str = "two-tower"


class PopularResponse(BaseModel):
    items: list[RecItem]
    latency_ms: float
    model: str = "popularity-baseline"


class SearchResponse(BaseModel):
    query: str
    items: list[RecItem]
    latency_ms: float
    source: str = "keyword"


class RecommendBatchRequest(BaseModel):
    """Batch recommend request (Section 14)."""

    user_id: str
    top_n: int = Field(default=10, ge=1, le=100)


class RankedItem(BaseModel):
    item_id: str
    score: float
    rank: int


class RecommendBatchResponse(BaseModel):
    """Batch recommend response (Section 14)."""

    user_id: str
    items: list[RankedItem]
    model_version: str


class EventRequest(BaseModel):
    """User interaction event (Section 14, 35)."""

    user_id: str
    item_id: str
    event_type: str
    timestamp: datetime | None = None


class EventResponse(BaseModel):
    status: str
    received_at: str
    event_type: str
    user_id: str
    item_id: str


class ChatRecommendRequest(BaseModel):
    """Conversational recommendation request (10A.3). A natural-language query AND an
    explicit user id (the LLM is never asked to guess or fabricate a user)."""
    user_id: str
    message: str
    top_k: int = 10


# ═══════════════════════════════════════════════════════════════
# State (loaded at startup)
# ═══════════════════════════════════════════════════════════════

class ArtifactIntegrityError(RuntimeError):
    """Raised at startup when embeddings/data artifacts are inconsistent.

    Fail-fast by design: serving a mismatched artifact caused production
    IndexError (bug: user index 5889 out of bounds for size 453).
    """


def _validate_artifacts(model: NumpyRetriever) -> None:
    """Fail-fast integrity checks between embeddings.npz and the serving data.

    Raises ArtifactIntegrityError (never auto-corrects, never falls back) when:
      - embedding row counts do not match id arrays
      - id arrays do not match the parquet datasets exactly
      - embedding dimensions are invalid or disagree
      - ids are duplicated
      - index mapping could go out of bounds
    """
    n_emb_users, n_emb_items = int(model.user_emb.shape[0]), int(model.item_emb.shape[0])
    n_id_users, n_id_items = len(model.user_ids), len(model.item_ids)
    n_ds_users, n_ds_items = len(state.user_id_map), state.n_items

    def _fail(artifact: str, expected: object, actual: object) -> None:
        raise ArtifactIntegrityError(
            f"Artifact integrity check failed: artifact={artifact} "
            f"expected={expected} actual={actual}"
        )

    if n_id_users != n_emb_users:
        _fail("user_ids length vs user_emb rows", n_emb_users, n_id_users)
    if n_id_items != n_emb_items:
        _fail("item_ids length vs item_emb rows", n_emb_items, n_id_items)
    if n_ds_users != n_emb_users:
        _fail("dataset users vs user_emb rows", n_emb_users, n_ds_users)
    if n_ds_items != n_emb_items:
        _fail("dataset items vs item_emb rows", n_emb_items, n_ds_items)

    if len(set(model.user_ids.tolist())) != n_id_users:
        _fail("unique user ids in npz", n_id_users, len(set(model.user_ids.tolist())))
    if len(set(model.item_ids.tolist())) != n_id_items:
        _fail("unique item ids in npz", n_id_items, len(set(model.item_ids.tolist())))

    emb_user_ids, emb_item_ids = set(map(str, model.user_ids)), set(map(str, model.item_ids))
    if emb_user_ids != set(state.user_id_map):
        missing = len(set(state.user_id_map) - emb_user_ids)
        extra = len(emb_user_ids - set(state.user_id_map))
        _fail("user id sets (dataset vs npz)", f"match (missing={missing})", f"extra={extra}")
    if emb_item_ids != set(state.item_id_map):
        missing = len(set(state.item_id_map) - emb_item_ids)
        extra = len(emb_item_ids - set(state.item_id_map))
        _fail("item id sets (dataset vs npz)", f"match (missing={missing})", f"extra={extra}")

    d_user, d_item = int(model.user_emb.shape[1]), int(model.item_emb.shape[1])
    if d_user <= 0 or d_item <= 0:
        _fail("embedding dimensions positive", ">0", f"user={d_user} item={d_item}")
    if d_user != d_item:
        _fail("user/item embedding dims equal", d_user, d_item)
    if not np.isfinite(model.user_emb).all() or not np.isfinite(model.item_emb).all():
        _fail("finite embedding values", "all finite", "NaN/Inf present")

    _verify_manifest(model)


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ids_sha256(ids: list[str] | Any) -> str:
    import hashlib

    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def _verify_manifest(model: NumpyRetriever) -> None:
    """Verify artifacts against the versioned manifest (P2).

    Migration policy: if the manifest is present it is verified STRICTLY
    (hash mismatch = fail-fast); if it has not been published yet we only warn,
    so the current release keeps starting while the manifest rollout completes.
    """
    emb_path = Path(MODEL_PATH)
    manifest_path = emb_path.parent / "manifest.json"
    if not manifest_path.exists() and MANIFEST_URL:
        _download(MANIFEST_URL, manifest_path)
    if not manifest_path.exists():
        print("[startup] WARNING: no artifact manifest found — skipping P2 hash checks")
        return
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        raise ArtifactIntegrityError(
            f"Artifact integrity check failed: artifact=manifest expected=valid JSON actual={e}"
        ) from None

    def _fail(artifact: str, expected: object, actual: object) -> None:
        raise ArtifactIntegrityError(
            f"Artifact integrity check failed: artifact={artifact} "
            f"expected={expected} actual={actual}"
        )

    schema = manifest.get("schema_version")
    if schema != 1:
        _fail("manifest schema_version", 1, schema)
    if int(manifest.get("user_count", -1)) != int(model.user_emb.shape[0]):
        _fail(
            "manifest user_count vs user_emb rows",
            model.user_emb.shape[0],
            manifest.get("user_count"),
        )
    if int(manifest.get("item_count", -1)) != int(model.item_emb.shape[0]):
        _fail(
            "manifest item_count vs item_emb rows",
            model.item_emb.shape[0],
            manifest.get("item_count"),
        )

    ids_hash_user = manifest.get("user_ids_sha256")
    if ids_hash_user and ids_hash_user != _ids_sha256(model.user_ids.tolist()):
        _fail("user_ids_sha256", ids_hash_user[:12] + "…", "mismatch")
    ids_hash_item = manifest.get("item_ids_sha256")
    if ids_hash_item and ids_hash_item != _ids_sha256(model.item_ids.tolist()):
        _fail("item_ids_sha256", ids_hash_item[:12] + "…", "mismatch")

    arts = manifest.get("artifacts", {})
    files = {
        "embeddings.npz": emb_path,
        "items.parquet": Path(DATA_DIR) / "items.parquet",
        "interactions_train.parquet": Path(DATA_DIR) / "interactions_train.parquet",
    }
    for name, path in files.items():
        meta = arts.get(name)
        if not meta or not path.exists():
            continue  # per-file entries are optional; missing file already failed earlier
        digest = _sha256_file(path)
        if digest != meta.get("sha256"):
            _fail(
                f"{name} sha256 (changed without manifest bump)",
                str(meta.get("sha256", ""))[:12] + "…",
                digest[:12] + "…",
            )
    print(f"[manifest] verified OK (release={manifest.get('release_version')})")


class AppState:
    model: NumpyRetriever | None = None
    model_version: str = MODEL_VERSION
    user_id_map: dict[str, int] = {}
    item_id_map: dict[str, int] = {}
    idx_to_item: dict[int, str] = {}
    items_df: pd.DataFrame | None = None
    train_df: pd.DataFrame | None = None
    n_items: int = 0
    device: str = "cpu"
    llm: LLMService | None = None
    events_buffer: deque[dict] = deque(maxlen=2000)
    qdrant_client: Any = None
    qdrant_available: bool = False
    qdrant_collection: str = QDRANT_COLLECTION
    embedder: Any = None


state = AppState()


def _init_qdrant() -> None:
    """Probe Qdrant at startup (non-fatal). Falls back to keyword search if down (Section 30)."""
    if not QDRANT_URL:
        print("[qdrant] QDRANT_URL not set, vector search disabled")
        return
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
            https=QDRANT_URL.startswith("https"),
        )
        client.get_collection(collection_name=QDRANT_COLLECTION)
        state.qdrant_client = client
        state.qdrant_available = True

        print(
            f"[qdrant] connected to {QDRANT_URL} (collection={QDRANT_COLLECTION}); "
            "sentence embedder initialized lazily on first /search"
        )
    except Exception as e:  # noqa: BLE001
        state.qdrant_available = False
        print(f"[qdrant] unavailable ({e}); falling back to keyword search")


def _get_embedder():
    """Load the ONNX MiniLM encoder lazily (kept off the startup path to keep
    the free-tier memory footprint low). Vector-compatible with the model used
    to index the Qdrant collection (same all-MiniLM-L6-v2 weights, 384-dim)."""
    if state.embedder is None:
        from app.semantic import get_encoder

        enc = get_encoder()
        if enc is None and SEMANTIC_ASSETS_URL:
            # Stateless cloud start: fetch tokenizer + onnx weights once.
            SEMANTIC_DIR = os.environ.get("SEMANTIC_ASSETS_DIR", "artifacts/semantic")
            _download(
            f"{SEMANTIC_ASSETS_URL}/tokenizer.json", Path(SEMANTIC_DIR) / "tokenizer.json"
        )
            _download(f"{SEMANTIC_ASSETS_URL}/model.onnx", Path(SEMANTIC_DIR) / "model.onnx")
            from app.semantic import get_encoder as _ge

            enc = _ge()
        state.embedder = enc
    return state.embedder


def _download(url: str, dest: Path) -> None:
    """Download a file with retries; used for stateless cloud startup."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
            print(f"[download] {dest} <- {url} ({dest.stat().st_size / 1e6:.1f} MB)")
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last_err}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data at startup; stateless deploy pulls them via URL if missing."""
    global state
    t0 = time.time()

    # ── Data files (items catalog + training interactions) ──
    items_path = Path(DATA_DIR) / "items.parquet"
    train_path = Path(DATA_DIR) / "interactions_train.parquet"
    if not items_path.exists() and DATA_BASE_URL:
        _download(f"{DATA_BASE_URL}/items.parquet", items_path)
    if not train_path.exists() and DATA_BASE_URL:
        _download(f"{DATA_BASE_URL}/interactions_train.parquet", train_path)
    if not items_path.exists() or not train_path.exists():
        raise RuntimeError(
            f"Missing data files in {DATA_DIR}. Run `make data` or set DATA_BASE_URL."
        )

    state.items_df = pd.read_parquet(items_path)
    state.train_df = pd.read_parquet(train_path)

    state.item_id_map = {iid: i for i, iid in enumerate(state.items_df["item_id"])}
    state.idx_to_item = {v: k for k, v in state.item_id_map.items()}
    state.n_items = len(state.item_id_map)

    unique_users = sorted(state.train_df["user_id"].unique())
    state.user_id_map = {uid: i for i, uid in enumerate(unique_users)}

    state.device = "cpu"  # NumPy serving path — no torch on the free tier
    # Eagerly raise if a configured prompt version is missing (10A.2) so startup fails fast.
    state.llm = LLMService(client=None)
    # (LLMService is intentionally None-client -> frank stub; never a SPOF, 10A.8.)

    # ── Qdrant vector search (optional, Section 11) ──
    _init_qdrant()

    # ── NumPy embeddings (stateless: EMBEDDINGS_URL fallback) ──
    emb_path = Path(MODEL_PATH)
    if not emb_path.exists() and MODEL_URL:
        _download(MODEL_URL, emb_path)

    if emb_path.exists():
        try:
            state.model = load_retriever(str(emb_path))
            # Fail-fast on artifact/dataset mismatch (bug: IndexError 5889 vs 453).
            _validate_artifacts(state.model)
            state.model_version = state.model.version
        except ArtifactIntegrityError:
            raise  # never serve a mismatched artifact
        except Exception as e:  # noqa: BLE001
            print(f"[startup] Embeddings load failed (falling back to untrained): {e}")

    if state.model is None:
        # Serve an untrained retriever (same behavior as the old torch fallback).
        rng = np.random.default_rng(42)
        emb_range = 0.5 / 64
        n_users = max(1, len(state.user_id_map))
        state.model = NumpyRetriever(
            user_emb=rng.uniform(-emb_range, emb_range, size=(n_users, 64)).astype(np.float32),
            item_emb=rng.uniform(-emb_range, emb_range, size=(state.n_items, 64)).astype(
                np.float32
            ),
            user_ids=np.array(sorted(state.user_id_map), dtype=np.str_),
            item_ids=np.array(list(state.item_id_map), dtype=np.str_),
            version="untrained",
        )
        state.model_version = "untrained"
        print(f"WARNING: No embeddings at {MODEL_PATH}, serving untrained model")

    print(f"Startup complete in {time.time() - t0:.2f}s "
          f"(users={len(state.user_id_map)}, items={state.n_items}, device={state.device})")
    yield


app = FastAPI(title="RecSys API", version="0.5.0", lifespan=lifespan)


@app.middleware("http")
async def http_metrics(request: Request, call_next):
    path = request.url.path
    method = request.method
    with HTTP_LATENCY.labels(method, path).time():
        response = await call_next(request)
    HTTP_REQUESTS.labels(method, path).inc()
    return response


# ═══════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "model_version": state.model_version,
        "device": state.device,
        "users": len(state.user_id_map),
        "items": state.n_items,
        "events_buffered": len(state.events_buffer),
        "qdrant_available": state.qdrant_available,
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend(
    user_id: str,
    k: int = Query(default=20, ge=1, le=100),
) -> RecommendResponse:
    """Return top-K recommendations for a known user."""
    t0 = time.time()

    if user_id not in state.user_id_map:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in training data")

    user_idx = state.user_id_map[user_id]
    REC_REQUESTS.labels("get_recommend").inc()

    # Build exclude mask from training interactions for this user
    if state.train_df is not None:
        user_interactions = state.train_df[state.train_df["user_id"] == user_id]
        exclude_seen = pd.DataFrame({
            "user_idx": [user_idx] * len(user_interactions),
            "item_idx": [state.item_id_map.get(iid, -1) for iid in user_interactions["item_id"]],
        })
        exclude_seen = exclude_seen.loc[exclude_seen["item_idx"] >= 0]
    else:
        exclude_seen = None

    if state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    excl = (
        exclude_seen["item_idx"].tolist()
        if exclude_seen is not None and len(exclude_seen)
        else None
    )
    ranked = state.model.recommend(
        user_idx, k=k, exclude_seen_item_idx=cast(list[int] | None, excl)
    )[:k]

    items = _indices_to_items(ranked)
    return RecommendResponse(
        user_id=user_id, items=items, latency_ms=round((time.time() - t0) * 1000, 2)
    )


@app.post("/recommend", response_model=RecommendBatchResponse)
async def recommend_batch(req: RecommendBatchRequest) -> RecommendBatchResponse:
    """Batch recommend endpoint (Section 14)."""
    time.time()

    if req.user_id not in state.user_id_map:
        raise HTTPException(
            status_code=404, detail=f"User {req.user_id} not found in training data"
        )

    user_idx = state.user_id_map[req.user_id]
    REC_REQUESTS.labels("post_recommend").inc()

    if state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    ranked = state.model.recommend(user_idx, k=req.top_n)[: req.top_n]

    items = [
        RankedItem(item_id=str(state.idx_to_item.get(idx, "")), score=float(score), rank=rank)
        for rank, (idx, score) in enumerate(ranked, start=1)
        if idx in state.idx_to_item
    ]
    return RecommendBatchResponse(
        user_id=req.user_id, items=items, model_version=state.model_version
    )


@app.post("/event", response_model=EventResponse)
async def log_event(req: EventRequest) -> EventResponse:
    """Log a user interaction event (Section 14/35).

    Batch 1: validated and buffered in-memory (+ Prometheus counter). The Redis/Kafka
    real-time pipeline consumes these events in Batch 2 (Phase 10).
    """
    event_type = req.event_type.lower()
    if event_type not in EVENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"event_type must be one of {sorted(EVENT_TYPES)}",
        )
    if req.item_id not in state.item_id_map:
        raise HTTPException(status_code=404, detail=f"Item {req.item_id} not found in catalog")

    now = datetime.now(UTC)
    record = {
        "user_id": req.user_id,
        "item_id": req.item_id,
        "event_type": event_type,
        "timestamp": (req.timestamp or now).isoformat(),
        "received_at": now.isoformat(),
    }
    state.events_buffer.append(record)
    EVENTS.labels(event_type).inc()

    if state.user_id_map.get(req.user_id) is None:
        print(f"[event] unknown user {req.user_id} (cold-start, still logged)")

    return EventResponse(
        status="ok",
        received_at=record["received_at"],
        event_type=event_type,
        user_id=req.user_id,
        item_id=req.item_id,
    )


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus-compatible metrics (Section 14, 35)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/popular", response_model=PopularResponse)
async def popular(k: int = Query(default=20, ge=1, le=100)) -> PopularResponse:
    """Return popular items (cold-start / fallback strategy)."""
    t0 = time.time()
    if state.train_df is None or state.items_df is None:
        raise HTTPException(status_code=503, detail="data not loaded")

    # Compute popularity scores: weighted count of interactions
    weights = build_weighted_counts(state.train_df)
    weighted_df = state.train_df.assign(weight=weights.values)
    grouped: pd.Series = cast(
        pd.Series, weighted_df.groupby("item_id")["weight"].sum()
    )
    item_scores = grouped.sort_values(ascending=False)
    score_by_id: dict = item_scores.to_dict()

    top_ids = item_scores.head(k).index.tolist()
    items = []
    for iid in top_ids:
        row = state.items_df.loc[state.items_df["item_id"] == iid]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        items.append(RecItem(
            item_id=str(iid), title=str(r["title"]), category=str(r["category"]),
            score=float(score_by_id[iid]),
        ))
    return PopularResponse(items=items, latency_ms=round((time.time() - t0) * 1000, 2))


@app.get("/search", response_model=SearchResponse)
async def search(q: str = Query(..., min_length=2, max_length=100)) -> SearchResponse:
    """Item search. Semantic (Qdrant) when available, else keyword fallback (Section 30).

    Query-time enrichment (no model / index change):
      1. Vietnamese movie queries are mapped to English equivalents before
         embedding (app.query_norm) — the response still echoes the ORIGINAL q.
      2. Semantic top-N candidates are re-ranked with capped metadata bonuses
         (category / title / tags / description — app.rerank); the semantic
         score stays dominant and `source` remains "semantic".
    """
    t0 = time.time()
    if state.items_df is None:
        raise HTTPException(status_code=503, detail="items catalog not loaded")

    norm = normalize_query(q)
    items: list[RecItem] = []
    source = "keyword"
    if state.qdrant_available and state.qdrant_client is not None:
        QDRANT_REQUESTS.inc()
        try:
            from training.embeddings import query_qdrant

            embedder = _get_embedder()
            enc_t0 = time.time()
            vector = embedder.encode([norm.normalized], convert_to_numpy=True)[0]
            ONNX_INFERENCE_SECONDS.observe(time.time() - enc_t0)
            qd_t0 = time.time()
            hits = query_qdrant(
                state.qdrant_client, state.qdrant_collection, vector, limit=50
            )
            QDRANT_LATENCY.observe(time.time() - qd_t0)
            for hit in hits:
                iid = str(hit.payload.get("item_id", "")) if hit.payload else ""
                if not iid or iid not in state.item_id_map:
                    continue
                row = state.items_df.loc[state.items_df["item_id"] == iid]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                items.append(RecItem(
                    item_id=iid, title=str(r["title"]),
                    category=str(r["category"]), score=float(hit.score),
                ))
            source = "semantic"
        except Exception as e:  # noqa: BLE001
            QDRANT_ERRORS.labels(error_type=_classify_qdrant_error(e)).inc()
            print(f"[search] semantic failed ({type(e).__name__}); keyword fallback")

    query_tokens = content_tokens(q)
    if items:
        # Metadata-aware re-rank over the semantic candidates.
        scored: list[tuple[float, RecItem]] = []
        for it in items:
            row = state.items_df.loc[state.items_df["item_id"] == it.item_id]
            tags = str(row.iloc[0].get("tags", "")) if len(row) else ""
            desc = str(row.iloc[0].get("text_description", "")) if len(row) else ""
            s = score_candidate(
                semantic=it.score, title=it.title, category=it.category,
                tags=tags, description=desc,
                query_tokens=query_tokens,
                detected_categories=norm.categories,
            )
            scored.append((s.final, it))
        scored.sort(key=lambda p: p[0], reverse=True)
        items = [RecItem(
            item_id=it.item_id, title=it.title, category=it.category,
            score=round(min(1.0, sc), 4),
        ) for sc, it in scored[:20]]
    elif source == "keyword":
        # Keyword fallback matches on BOTH the raw key and the normalized
        # English tokens, so Vietnamese queries still find genre matches.
        norm_tokens = content_tokens(norm.normalized)
        title_l = cast(pd.Series, state.items_df["title"]).map(
            lambda t: strip_accents(str(t)).lower())
        tags_l = cast(pd.Series, state.items_df["tags"]).map(
            lambda t: strip_accents(str(t)).lower())
        desc_l = (
            cast(pd.Series, state.items_df["text_description"]).map(
                lambda t: strip_accents(str(t)).lower())
            if "text_description" in state.items_df.columns
            else pd.Series("", index=state.items_df.index)
        )
        q_key = strip_accents(q).lower()
        phrase_mask = (
            title_l.str.contains(q_key, regex=False)
            | tags_l.str.contains(q_key, regex=False)
            | desc_l.str.contains(q_key, regex=False)
        )
        cat_l = cast(pd.Series, state.items_df["category"]).astype(str).map(
            lambda c: strip_accents(c).lower())
        token_mask = pd.Series(False, index=state.items_df.index)
        for tok in norm_tokens:
            token_mask |= (
                title_l.str.contains(tok, regex=False)
                | tags_l.str.contains(tok, regex=False)
                | cat_l.str.contains(tok, regex=False)
                | desc_l.str.contains(tok, regex=False)
            )
        mask = phrase_mask | token_mask
        results = state.items_df.loc[mask].copy()
        # Keyword path ordering: explicit category signal first, then title hits.
        cat_rank = results["category"].map(
            lambda c: next(
                (i for i, cat in enumerate(norm.categories)
                 if str(c).strip().lower() == cat.lower()), 9))
        results = results.assign(_cr=cat_rank).sort_values("_cr").head(20)
        for _, r in results.iterrows():
            items.append(RecItem(
                item_id=str(r["item_id"]), title=str(r["title"]),
                category=str(r["category"]), score=1.0,
            ))

    resp = SearchResponse(
        query=q, items=items, latency_ms=round((time.time() - t0) * 1000, 2),
        source=source,
    )
    # One primary outcome per request, derived from the served source.
    if source == "semantic":
        SEMANTIC_SEARCH_TOTAL.inc()
    else:
        KEYWORD_FALLBACK_TOTAL.inc()
    return resp


# ═══════════════════════════════════════════════════════════════
class ChatRecommendResponse(BaseModel):
    user_id: str
    message: str
    items: list[RecItem]
    latency_ms: float
    llm_mode: str
    prompt_version: int


def _run_engine(user_id: int, k: int) -> list[RecItem]:
    """Shared Two-Tower retrieval (single user) -> RecItem list. Returns [] if model is None."""
    if state.model is None:
        return []
    ranked = state.model.recommend(user_id, k=k)[:k]
    return _indices_to_items(list(ranked))


@app.post("/chat/recommend", response_model=ChatRecommendResponse)
async def chat_recommend(req: ChatRecommendRequest) -> ChatRecommendResponse:
    """LLOMM H5: LLM is only the understanding layer. Parse the request, run the engine,
    optionally explain, then answer conversationally. LLM out/down => structured response
    with llm_mode=fallback (LLM is never a SPOF, 10A.8)."""
    t0 = time.time()
    uid = state.user_id_map.get(req.user_id)

    # 1) LLM understands the request -> a StructuredFilter (10A.1). Optional layer.
    filter_ = None
    if state.llm is not None:
        try:
            filter_ = state.llm.extract_preference(req.message)
        except Exception:  # noqa: BLE001
            filter_ = None

    # 2) always run the engine (LLM only translates; it never picks items).
    engine_items = _run_engine(uid, req.top_k) if uid is not None else []

    # 3) degrade: filter the catalog by what we understood (fallback to text match).
    if not engine_items:
        df = state.items_df
        if df is not None and not df.empty:
            mask = pd.Series(True, index=df.index)
            if filter_ is not None and filter_.category:
                mask &= df["category"].astype(str).str.lower().eq(filter_.category.lower())
            if filter_ is not None and filter_.color:
                mask &= df["tags"].astype(str).str.lower().str.contains(filter_.color.lower())
            r = df.loc[mask].head(req.top_k)
            items = [
                RecItem(
                    item_id=str(r2["item_id"]), title=str(r2["title"]),
                    category=str(r2["category"]), score=1.0,
                )
                for _, r2 in r.iterrows()
            ]
        else:
            items = []
    else:
        items = engine_items

    # 4) conversational + grounding. LLM never invents.
    llm_mode = (
        "ok"
        if (filter_ is not None and state.llm is not None)
        else "fallback"
    )
    prompt_version = state.llm.prompt_version if state.llm is not None else 1

    return ChatRecommendResponse(
        user_id=req.user_id,
        message=req.message,
        items=items,
        latency_ms=round((time.time() - t0) * 1000, 2),
        llm_mode=llm_mode,
        prompt_version=prompt_version,
    )


# Frontend (static files)
# ═══════════════════════════════════════════════════════════════

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/app", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    with open(STATIC_DIR / "index.html") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _indices_to_items(indices: list[tuple[int, float]] | list[int]) -> list[RecItem]:
    """Convert internal (item_idx, score) pairs (or bare indices) to RecItem list."""
    if state.items_df is None:
        return []
    items = []
    for entry in indices:
        idx, score = entry if isinstance(entry, tuple) else (entry, 0.0)
        iid = state.idx_to_item.get(idx)
        if iid is None:
            continue
        row = state.items_df.loc[state.items_df["item_id"] == iid]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        items.append(RecItem(
            item_id=iid, title=str(r["title"]), category=str(r["category"]), score=float(score)
        ))
    return items
