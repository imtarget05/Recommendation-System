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

import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from llm.service import LLMService
from training.common import build_weighted_counts
from training.two_tower import TwoTowerModel, retrieve_top_k_with_scores

DATA_DIR = os.environ.get("DATA_DIR", "data/processed")
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/two_tower.pt")
MODEL_URL = os.environ.get("MODEL_URL", "")
DATA_BASE_URL = os.environ.get("DATA_BASE_URL", "")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "two-tower-v1")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
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

class AppState:
    model: TwoTowerModel | None = None
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

        from sentence_transformers import SentenceTransformer

        state.embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[qdrant] connected to {QDRANT_URL} (collection={QDRANT_COLLECTION})")
    except Exception as e:  # noqa: BLE001
        state.qdrant_available = False
        print(f"[qdrant] unavailable ({e}); falling back to keyword search")


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

    state.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Eagerly raise if a configured prompt version is missing (10A.2) so startup fails fast.
    state.llm = LLMService(client=None)
    # (LLMService is intentionally None-client -> frank stub; never a SPOF, 10A.8.)

    # ── Qdrant vector search (optional, Section 11) ──
    _init_qdrant()

    # ── Trained checkpoint (stateless: MODEL_URL fallback) ──
    checkpoint_path = Path(MODEL_PATH)
    if not checkpoint_path.exists() and MODEL_URL:
        _download(MODEL_URL, checkpoint_path)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=state.device, weights_only=False)
        state.model = TwoTowerModel(
            n_users=len(state.user_id_map),
            n_items=state.n_items,
            emb_dim=checkpoint.get("emb_dim", 64),
        )
        state.model.load_state_dict(checkpoint["model_state"])
        state.model.to(state.device)
        state.model.eval()
        state.model_version = checkpoint.get("version", MODEL_VERSION)
    else:
        # Use an untrained model (will still serve, but metrics will be random)
        state.model = TwoTowerModel(
            n_users=len(state.user_id_map),
            n_items=state.n_items,
            emb_dim=64,
        )
        state.model.to(state.device)
        state.model.eval()
        state.model_version = "untrained"
        print(f"WARNING: No checkpoint at {MODEL_PATH}, serving untrained model")

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

    ranked = retrieve_top_k_with_scores(
        state.model, [user_idx], state.n_items, k=k,
        device=state.device, exclude_seen=cast(pd.DataFrame | None, exclude_seen),
    )[user_idx][:k]

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

    ranked = retrieve_top_k_with_scores(
        state.model, [user_idx], state.n_items, k=req.top_n, device=state.device
    )[user_idx][: req.top_n]

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
    """Item search. Semantic (Qdrant) when available, else keyword fallback (Section 30)."""
    t0 = time.time()
    if state.items_df is None:
        raise HTTPException(status_code=503, detail="items catalog not loaded")

    items: list[RecItem] = []
    source = "keyword"
    if state.qdrant_available and state.embedder is not None and state.qdrant_client is not None:
        try:
            from training.embeddings import query_qdrant

            vector = state.embedder.encode([q], convert_to_numpy=True)[0]
            hits = query_qdrant(
                state.qdrant_client, state.qdrant_collection, vector, limit=20
            )
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
            print(f"[search] semantic failed ({e}); keyword fallback")

    if not items:
        q_lower = q.lower()
        mask = (
            cast(pd.Series, state.items_df["title"]).str.lower().str.contains(q_lower)
            | cast(pd.Series, state.items_df["tags"]).str.lower().str.contains(q_lower)
        )
        results = state.items_df.loc[mask].head(20)
        for _, r in results.iterrows():
            items.append(RecItem(
                item_id=str(r["item_id"]), title=str(r["title"]),
                category=str(r["category"]), score=1.0,
            ))

    resp = SearchResponse(
        query=q, items=items, latency_ms=round((time.time() - t0) * 1000, 2),
        source=source,
    )
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
    ranked = retrieve_top_k_with_scores(
        state.model, [user_id], state.n_items, k=k,
        device=state.device, exclude_seen=None,
    )[user_id][:k]
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
