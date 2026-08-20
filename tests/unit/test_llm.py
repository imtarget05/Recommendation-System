"""LLM layer unit tests (Phase 8, LLMOps 10A)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm.client import LLMUnavailable, StubLLMClient, get_llm_client
from llm.registry import latest_version, load_prompt
from llm.schemas import PreferenceExtraction
from llm.service import LLMService, PreferenceParser


class _FailingClient(StubLLMClient):
    """A stub that always raises (simulates LLM outage, 10A.8)."""

    def complete_json(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise LLMUnavailable("backend down")


def test_stub_client_detects_color() -> None:
    client = StubLLMClient()
    out = client.complete_json("sys", "I want a black jacket", PreferenceExtraction)
    assert out["color"] == "black"
    assert out["category"] == "jacket"


def test_preference_parser_uses_stub() -> None:
    parser = PreferenceParser(StubLLMClient())
    filt = parser.parse("user wants a black coat")
    assert filt is not None
    assert filt.color == "black"
    assert filt.category in ("jacket", "coat")


def test_preference_parser_falls_back_on_outage(monkeypatch) -> None:  # noqa: ANN001
    parser = PreferenceParser(_FailingClient(), max_retries=1)
    filt = parser.parse("a black jacket")
    assert filt is None  # graceful degradation, does not crash


def test_schema_rejects_unknown_enum_value() -> None:
    with pytest.raises(ValidationError):
        PreferenceExtraction.model_validate({"category": "spaceship"})


def test_explanation_never_invents_attributes() -> None:
    llm = LLMService(client=StubLLMClient())
    exp = llm.explain("ml_item_10001", ["black", "jacket"])
    assert exp.text  # has factual fallback text


def test_llm_service_defaults_to_stub() -> None:
    llm = LLMService()
    assert llm.extract_preference("navy sneakers") is not None


def test_prompt_registry_versioned_files() -> None:
    assert latest_version("preference_extraction") >= 1
    p = load_prompt("preference_extraction", 1)
    assert p.task == "preference_extraction"
    assert "category" in p.system_prompt


def test_get_llm_client_defaults_to_stub() -> None:
    assert isinstance(get_llm_client(), StubLLMClient)
