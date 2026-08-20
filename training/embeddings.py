"""Embedding system (Section 7, §8 embedding pipeline).

Item embedding → Qdrant vector store. User embedding = weighted sum of item embeddings
from interaction history. Batch generation + upsert.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from training.common import load_processed

ITEM_TEXT_FIELDS = ["title", "category", "tags", "text_description"]


def build_item_text(row: pd.Series) -> str:
    """Construct semantic text from item metadata (Section 7.2)."""
    parts = []
    for field in ITEM_TEXT_FIELDS:
        val = row.get(field)
        if val is not None and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def encode_items(items_df: pd.DataFrame, model) -> np.ndarray:
    """Encode item text into vectors (batch encoded for efficiency)."""
    texts = items_df.apply(build_item_text, axis=1).tolist()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors


def build_user_embedding(user_id: str, interactions: pd.DataFrame, item_vectors: np.ndarray,
                         item_index: dict) -> np.ndarray:
    """Compute user embedding = weighted sum of item vectors from interaction history.

    Weight = event_weight × recency_weight (Section 6.3).
    """
    user_interactions = cast(pd.DataFrame, interactions[interactions["user_id"] == user_id])
    if len(user_interactions) == 0:
        dim = item_vectors.shape[1] if item_vectors.ndim == 2 else item_vectors.size
        return np.zeros(dim, dtype=np.float32)

    from training.common import build_weighted_counts
    weights = build_weighted_counts(user_interactions)

    # Map item_id to vector index
    indices = np.array([item_index.get(iid, -1) for iid in user_interactions["item_id"]])
    valid = indices >= 0
    if not valid.any():
        dim = item_vectors.shape[1] if item_vectors.ndim == 2 else item_vectors.size
        return np.zeros(dim, dtype=np.float32)

    valid_indices = indices[valid]
    valid_weights = weights[valid]

    # Weighted sum divided by total weight
    valid_item_vectors = item_vectors[valid_indices]
    summed = (valid_item_vectors * valid_weights[:, np.newaxis]).sum(axis=0)
    total_weight = valid_weights.sum()
    if total_weight == 0:
        dim = item_vectors.shape[1] if item_vectors.ndim == 2 else item_vectors.size
        return np.zeros(dim, dtype=np.float32)
    return (summed / total_weight).astype(np.float32)


def collection_exists(client, collection_name: str) -> bool:
    """Check if a Qdrant collection exists."""
    try:
        client.get_collection(collection_name=collection_name)
        return True
    except Exception:
        return False


def init_qdrant_collection(client, collection_name: str, vector_size: int) -> None:
    """Create Qdrant collection if it does not exist."""
    if collection_exists(client, collection_name):
        return

    from qdrant_client.http import models as qmodels

    client.create_collection(
        collection_name=collection_name,
        hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=40),
        vectors_config=qmodels.VectorParams(
            size=vector_size, distance=qmodels.Distance.COSINE
        ),
    )


def upsert_to_qdrant(client, collection_name: str, item_vectors: np.ndarray,
                     item_ids: list[str], payloads: list[dict]) -> None:
    """Batch upsert into Qdrant collection (minimal payloads).

    Strategy: only keep 'item_id' in payload to stay under Qdrant's 33MB limit.
    """
    points = []
    for iid, vec, payload in zip(item_ids, item_vectors, payloads):
        # ABSOLUTE MINIMAL: only item_id; drop everything else to avoid >33MB limit
        points.append(
            {
                "id": iid,
                "vector": vec.tolist(),
                "payload": {"item_id": iid},
            }
        )
    client.upsert(collection_name=collection_name, points=points)


def query_qdrant(client, collection_name: str, vector: np.ndarray, limit: int = 10):
    """Query Qdrant for similar items."""
    result = client.search(
        collection_name=collection_name,
        query_vector=vector.tolist(),
        limit=limit,
        with_payload=["item_id"],
    )
    return result


def run_embedding_pipeline(
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    """Full pipeline: load data → encode → init Qdrant → upsert (item_id-only payloads).

    Returns dict with generation stats.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    data = load_processed()
    items_df = data["items"]

    # Load model
    model = SentenceTransformer(model_name)

    # Encode items
    item_texts = items_df.apply(build_item_text, axis=1)
    item_vectors = model.encode(item_texts.tolist(), convert_to_numpy=True)
    vector_size = item_vectors.shape[1]
    item_ids = items_df["item_id"].tolist()

    # Minimal payloads: item_id only (prevents >33MB JSON limit)
    payloads = [{"item_id": iid} for iid in item_ids]

    # Qdrant setup
    client = QdrantClient(host="localhost", port=6333)

    # Create collection (will overwrite if exists via delete+create handled by caller)
    client.create_collection(
        collection_name="items",
        hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=40),
        vectors_config=qmodels.VectorParams(
            size=vector_size, distance=qmodels.Distance.COSINE
        ),
    )

    # Upsert
    upsert_to_qdrant(client, "items", item_vectors, item_ids, payloads)

    # Verify
    info = client.get_collection(collection_name="items")
    indexed = info.indexed_vectors_count or info.points_count

    return {
        "n_items": len(item_vectors),
        "vector_size": vector_size,
        "indexed_in_qdrant": indexed,
        "model": model_name,
    }
