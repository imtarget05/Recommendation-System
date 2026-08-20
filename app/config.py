from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Central list of env vars that override config.yaml (Section 28)
_ENV_OVERRIDES = {
    "DATASET_NAME": "data.default_dataset",
    "MAX_ROWS": "data.max_rows",
    "LLM_PROVIDER": "llm.provider",
    "LLM_MODEL": "llm.model",
    "OLLAMA_URL": "llm.base_url",
    "LLM_MAX_RETRIES": "llm.max_retries",
    "LLM_PROMPT_VERSION": "llm.prompt_version",
    "LLM_CACHE_TTL": "llm.cache_ttl",
    "QDRANT_URL": "vectordb.url",
    "QDRANT_API_KEY": "vectordb.api_key",
    "QDRANT_COLLECTION": "vectordb.collection",
    "EMBEDDING_MODEL": "vectordb.embedding_model",
    "REDIS_URL": "realtime.redis_url",
    "KAFKA_BROKER_URL": "realtime.kafka_broker_url",
}

_DATA_DIR_SUBDIRS = {
    "raw_dir": "raw",
    "processed_dir": "processed",
    "features_dir": "features",
    "embeddings_dir": "embeddings",
    "manifests_dir": "manifests",
}

_SUBKEY_OVERRIDES = {
    "MOVIELENS_VERSION": "connectors.movielens.version",
    "MIN_INTERACTIONS_PER_USER": "data.min_interactions_per_user",
}


class InteractionWeights(BaseModel):
    view: float = 1.0
    click: float = 2.0
    like: float = 3.0
    purchase: float = 5.0


class RecencyConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    lambda_: float = Field(default=0.035, alias="lambda")
    unit: str = "days"
    max_age_days: float = 365.0


class SplitConfig(BaseModel):
    train: float = 0.7
    validation: float = 0.1
    test: float = 0.2


class DataDirs(BaseModel):
    raw_dir: Path = Path("./data/raw")
    processed_dir: Path = Path("./data/processed")
    features_dir: Path = Path("./data/features")
    embeddings_dir: Path = Path("./data/embeddings")
    manifests_dir: Path = Path("./data/manifests")
    default_dataset: str = "movielens"
    max_rows: int = 5_000_000
    min_interactions_per_user: int = 2


class MovielensConnector(BaseModel):
    version: str = "ml-25m"
    base_url: str = "https://files.grouplens.org/datasets/movielens"
    like_rating_threshold: float = 4.0


class HMConnector(BaseModel):
    kaggle_ref: str = "michaelacook/h-and-m-personalized-fashion-recommendations"


class Connectors(BaseModel):
    movielens: MovielensConnector = MovielensConnector()
    hm: HMConnector = HMConnector()


class LLMConfig(BaseModel):
    """LLM layer config (Phase 8, LLMOps 10A). arg > env > yaml > default."""
    provider: str = "stub"
    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434"
    prompt_version: int = 1
    max_retries: int = 1
    cache_enabled: bool = True
    cache_ttl: float = 300.0
    token_budget: int = 4096
    fallback: str = "stub"


class MonitoringConfig(BaseModel):
    """LLM + serving observability switches (10A.5)."""
    drift_threshold: float = 0.05
    metrics_window: int = 1000
    sampling_rate: float = 1.0
    log_llm_latency: bool = True


class VectorDbConfig(BaseModel):
    """Qdrant vector search config (Phase 5, Section 11)."""
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection: str = "items"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384


class RealTimeConfig(BaseModel):
    """Reserved for Phase 10 real-time pipeline (Redis/Kafka)."""
    redis_url: str | None = None
    kafka_broker_url: str | None = None
    kafka_events_topic: str = "recsys-events"


class Config(BaseModel):
    data: DataDirs = DataDirs()
    interaction_weights: InteractionWeights = InteractionWeights()
    recency: RecencyConfig = RecencyConfig()
    split: SplitConfig = SplitConfig()
    connectors: Connectors = Connectors()
    llm: LLMConfig = LLMConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    vectordb: VectorDbConfig = VectorDbConfig()
    realtime: RealTimeConfig = RealTimeConfig()


def _apply_env_overrides(cfg: dict) -> dict:  # noqa: ANN001
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        data_cfg = cfg.setdefault("data", {})
        for field, sub in _DATA_DIR_SUBDIRS.items():
            data_cfg[field] = str(Path(data_dir) / sub)
    for env_key, path in {**_ENV_OVERRIDES, **_SUBKEY_OVERRIDES}.items():
        value = os.environ.get(env_key)
        if value is None:
            continue
        keys = path.split(".")
        target = cfg
        for part in keys[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        if isinstance(value, str) and value.replace(".", "", 1).isdigit() and value != "":
            value = float(value) if "." in value else int(value)
        target[keys[-1]] = value
    return cfg


@lru_cache
def load_config() -> Config:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    raw = yaml.safe_load(config_path.read_text()) or {}
    raw = _apply_env_overrides(raw)
    return Config(**raw)
