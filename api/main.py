"""Real-time serving API for Two-Tower recommendation system.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                — frontend HTML page
    GET  /health          — liveness probe
    GET  /recommend/{user_id} — top-K recommendations for a known user
    GET  /popular         — popular items (cold-start fallback)
    GET  /search?q=...    — keyword search over item titles
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm.service import LLMService
from training.common import build_weighted_counts
from training.two_tower import TwoTowerModel, retrieve_top_k

DATA_DIR = "data/processed"
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/two_tower.pt")


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
    user_id_map: dict[str, int] = {}
    item_id_map: dict[str, int] = {}
    idx_to_item: dict[int, str] = {}
    items_df: pd.DataFrame | None = None
    train_df: pd.DataFrame | None = None
    n_items: int = 0
    device: str = "cpu"
    llm: LLMService | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data at startup."""
    global state
    t0 = time.time()

    # Load item catalog
    state.items_df = pd.read_parquet(f"{DATA_DIR}/items.parquet")
    state.train_df = pd.read_parquet(f"{DATA_DIR}/interactions_train.parquet")

    state.item_id_map = {iid: i for i, iid in enumerate(state.items_df["item_id"])}
    state.idx_to_item = {v: k for k, v in state.item_id_map.items()}
    state.n_items = len(state.item_id_map)

    # Load user map from training data
    unique_users = sorted(state.train_df["user_id"].unique())
    state.user_id_map = {uid: i for i, uid in enumerate(unique_users)}

    state.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Eagerly raise if a configured prompt version is missing (10A.2) so startup fails fast.
    state.llm = LLMService(client=None)
    # (LLMService is intentionally None-client -> frank stub; never a SPOF, 10A.8.)

    # Load model
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=state.device, weights_only=False)
        state.model = TwoTowerModel(
            n_users=len(state.user_id_map),
            n_items=state.n_items,
            emb_dim=checkpoint.get("emb_dim", 64),
        )
        state.model.load_state_dict(checkpoint["model_state"])
        state.model.to(state.device)
        state.model.eval()
    else:
        # Use an untrained model (will still serve, but metrics will be random)
        state.model = TwoTowerModel(
            n_users=len(state.user_id_map),
            n_items=state.n_items,
            emb_dim=64,
        )
        state.model.to(state.device)
        state.model.eval()
        print(f"WARNING: No checkpoint at {MODEL_PATH}, serving untrained model")

    print(f"Startup complete in {time.time() - t0:.2f}s "
          f"(users={len(state.user_id_map)}, items={state.n_items}, device={state.device})")
    yield


app = FastAPI(title="RecSys API", version="0.5.0", lifespan=lifespan)


# ═══════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": state.model is not None, "device": state.device}


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

    rec_indices = retrieve_top_k(
        state.model, [user_idx], state.n_items, k=k,
        device=state.device, exclude_seen=cast(pd.DataFrame | None, exclude_seen),
    )[user_idx][:k]

    items = _indices_to_items(rec_indices)
    return RecommendResponse(
        user_id=user_id, items=items, latency_ms=round((time.time() - t0) * 1000, 2)
    )


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
    """Keyword search over item titles and tags."""
    t0 = time.time()
    q_lower = q.lower()
    if state.items_df is None:
        raise HTTPException(status_code=503, detail="items catalog not loaded")

    mask = (
        cast(pd.Series, state.items_df["title"]).str.lower().str.contains(q_lower)
        | cast(pd.Series, state.items_df["tags"]).str.lower().str.contains(q_lower)
    )
    results = state.items_df.loc[mask].head(20)

    items = []
    for _, r in results.iterrows():
        items.append(RecItem(
            item_id=str(r["item_id"]), title=str(r["title"]),
            category=str(r["category"]), score=1.0,
        ))
    return SearchResponse(query=q, items=items, latency_ms=round((time.time() - t0) * 1000, 2))


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
    rec_indices = retrieve_top_k(
        state.model, [user_id], state.n_items, k=k,
        device=state.device, exclude_seen=None,
    )[user_id][:k]
    return _indices_to_items(list(rec_indices))


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

def _indices_to_items(indices: list[int]) -> list[RecItem]:
    """Convert internal item indices to RecItem list."""
    if state.items_df is None:
        return []
    items = []
    for idx in indices:
        iid = state.idx_to_item.get(idx)
        if iid is None:
            continue
        row = state.items_df.loc[state.items_df["item_id"] == iid]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        items.append(RecItem(
            item_id=iid, title=str(r["title"]), category=str(r["category"]), score=0.0
        ))
    return items
