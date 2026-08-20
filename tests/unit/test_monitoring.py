"""Tests for monitoring + LLM observability metrics (Sections 19, 20, 10A.5-10A.8)."""

from __future__ import annotations

import numpy as np

from llm.client import StubLLMClient
from llm.service import CircuitBreaker, LLMService, PreferenceParser
from monitoring.metrics import (
    catalog_coverage,
    ctr,
    fallback_rate,
    intra_list_diversity,
    latency_percentiles,
    list_novelty,
    parse_failure_rate,
    personalization,
    rate,
    token_efficiency,
)


def test_catalog_coverage() -> None:
    assert catalog_coverage([1, 2, 2, 3], catalog_size=10) == 0.3


def test_intra_list_diversity_orthogonal() -> None:
    m = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    id2i = {"a": 0, "b": 1, "c": 2}
    assert intra_list_diversity(["a", "b"], id2i, m) > 0.9


def test_novelty_ranks_popular_lower() -> None:
    pop = {"a": 0.9, "b": 0.05, "c": 0.05}
    assert list_novelty(["a"], pop) < list_novelty(["b"], pop)


def test_personalization_identical_lists_low() -> None:
    m = np.eye(3)
    id2i = {"a": 0, "b": 1, "c": 2}
    p = personalization([["a", "b"], ["a", "b"]], id2i, m)
    assert p < 0.2


def test_ctr() -> None:
    assert ctr(0, 0) == 0.0
    assert ctr(5, 100) == 0.05


def test_latency_percentiles() -> None:
    d = latency_percentiles([10, 20, 30, 40, 50, 100], qs=(50, 95, 99))
    assert d["p50"] == 35.0
    assert d["p95"] == 87.5
    assert d["p99"] == 97.5
    assert latency_percentiles([])["p99"] == 0.0


def test_rate_zero_denominator() -> None:
    assert rate(0, 0) == 0.0
    assert rate(5, 0) == 0.0


def test_token_efficiency() -> None:
    d = token_efficiency(100, 50, budget_tokens=100000)
    assert d["tokens_used"] == 150
    assert d["tokens_remaining"] == 99850
    assert d["utilization"] == 0.0015


def test_parse_failure_rate() -> None:
    parser = PreferenceParser(StubLLMClient(), cache_enabled=False)
    parser.total_requests = 3
    parser.parse_failures = 1
    assert parse_failure_rate(parser) == 1 / 3


def test_fallback_rate_open_breaker() -> None:
    parser = PreferenceParser(StubLLMClient(), cache_enabled=False)
    parser.breaker = CircuitBreaker(threshold=1, cooldown=99999)
    parser.breaker.record_failure()  # opens the breaker
    assert parser.parse("a query") is None
    assert fallback_rate(parser) == 1.0


def test_parser_caches_results() -> None:
    parser = PreferenceParser(StubLLMClient(), cache_enabled=True)
    first = parser.parse("black coat")
    second = parser.parse("black coat")
    assert first == second
    assert len(parser.latency_ms) == 1  # second hit served from cache


def test_service_prefers_explicit_client() -> None:
    svc = LLMService(client=StubLLMClient())
    assert svc.extract_preference("navy sneakers") is not None
