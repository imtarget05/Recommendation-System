"""LLM layer package (Section 9-10, LLMOps 10A).

Keep import-light: no heavy model is loaded at import time. The orchestrator and
client are constructed lazily by :func:`llm.get_llm_service`.
"""

from __future__ import annotations

from .client import LLMClient, OllamaClient, StubLLMClient, get_llm_client
from .schemas import (
    Explanation,
    ItemFilter,
    ItemPrediction,
    PreferenceExtraction,
    StructuredFilter,
)
from .service import ExplanationBuilder, LLMService, PreferenceParser

__all__ = [
    "LLMClient",
    "OllamaClient",
    "StubLLMClient",
    "get_llm_client",
    "LLMService",
    "PreferenceParser",
    "ExplanationBuilder",
    "PreferenceExtraction",
    "StructuredFilter",
    "Explanation",
    "ItemPrediction",
    "ItemFilter",
]
