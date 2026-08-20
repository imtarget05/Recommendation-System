"""LLM provider client (Section 9, LLMOps 10A).

A thin provider abstraction so the recommendation/explanation code never talks to a
specific vendor. The local default targets an Ollama endpoint (Qwen3/Gemma) but the
`client` is swappable (spec rule: adapters/interfaces over hard-coded providers).

Only a lightweight HTTP call is made here for the real provider; heavy local model
loading is left to Ollama. A :class:`StubLLMClient` is provided for tests and for
systems without the model — it never hallucinates (returns empty/filtered results).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

from pydantic import BaseModel

try:  # the real client is optional so tests / light installs run without it
    import httpx  # noqa: F401  (availability probe)

    _HAS_HTTPX = True
except ImportError:  # pragma: no cover
    _HAS_HTTPX = False


class LLMClient(ABC):
    """Common interface for any LLM provider."""

    @abstractmethod
    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> dict:
        """Return a structured (JSON-decodable) response validated against `schema`.

        Raises a subclass of :class:`LLMUnavailable` on unreachable/failure.
        """


class LLMUnavailable(RuntimeError):
    """Raised when the LLM backend cannot be reached (LLMOps 10A.8)."""


class OllamaClient(LLMClient):
    """Minimal Ollama-compatible OpenAI-style client (local for free-tier demo)."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "qwen3:8b")

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> dict:
        if not _HAS_HTTPX:
            raise LLMUnavailable("httpx not installed; configure a real client or use the stub")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "options": {"temperature": temperature},
            "stream": False,
        }
        try:
            resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=30.0)
            resp.raise_for_status()
            body = resp.json()
            if "message" in body and "content" in body["message"]:
                text = body["message"]["content"]
            else:
                text = str(body)
            return json.loads(text)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:  # noqa: PERF203
            raise LLMUnavailable(f"ollama call failed: {exc}") from exc


class StubLLMClient(LLMClient):
    """Deterministic offline stub. Does not require a real model nor network.

    It returns structured data populated from the user prompt itself (hand-coded
    rule), so tests and the no-API demo still exercise the full pipeline.
    """

    def __init__(self, fallback_map: dict | None = None) -> None:
        self.fallback_map = fallback_map or {}

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> dict:
        # Extremely naive rule: if a color word appears, return it; category default
        # to "other". This makes the conversational flow deterministic in tests.
        text = f"{system_prompt} {user_prompt}".lower()
        color = None
        for c in ("black", "white", "red", "blue", "green", "beige", "grey", "navy"):
            if c in text:
                color = c
                break
        return {"category": "jacket", "color": color, "style": None, "price_preference": None}


def get_llm_client(
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    fallback_map: dict | None = None,
) -> LLMClient:
    """Build the configured provider (stub vs live). Provider is from env or arg."""
    provider = provider or os.getenv("LLM_PROVIDER", "stub")
    if provider == "ollama":
        return OllamaClient(base_url=base_url, model=model)
    return StubLLMClient(fallback_map=fallback_map)
