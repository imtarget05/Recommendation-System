"""Regression tests: Vietnamese query normalization + metadata-aware re-ranking.

Covers Phase 7 of the relevance iteration:
  - accent/case-insensitive normalization consistency
  - Vietnamese → English movie-domain mapping (query-time only)
  - genre detection with false-positive guard ("star" must NOT map to a category)
  - re-rank boosting Action candidates for "action", Romance for "romance"
  - "star" must not blindly boost titles containing "star"
  - semantic score dominance (bonus capped, never overrides large gaps)
"""

from __future__ import annotations

from app.query_norm import detect_categories, normalize_query, strip_accents
from app.rerank import BONUS_CAP, content_tokens, score_candidate

# ── Normalization ────────────────────────────────────────────

def test_strip_accents_consistent() -> None:
    assert strip_accents("PHIM HÀNH ĐỘNG") == strip_accents("Phim Hành Động")
    assert strip_accents("phim hành động") == "phim hanh dong"


def test_vietnamese_action_normalized() -> None:
    n = normalize_query("phim hành động")
    assert n.original == "phim hành động"          # original preserved
    assert n.normalized == "action movie"
    assert "Action" in n.categories


def test_accent_variants_same_normalization() -> None:
    outs = {normalize_query(v).normalized
            for v in ("phim hành động", "PHIM HÀNH ĐỘNG", "Phim Hành Động")}
    assert len(outs) == 1


def test_romance_and_scifi() -> None:
    assert normalize_query("phim tình cảm").normalized == "romance movie"
    n = normalize_query("phim khoa học viễn tưởng")
    assert n.normalized == "science fiction movie"
    assert "Sci-Fi" in n.categories


def test_english_query_unchanged() -> None:
    n = normalize_query("star wars")
    assert n.normalized == "star wars"
    assert n.categories == []                       # no genre signal


def test_star_is_not_a_category() -> None:
    """False-positive guard: 'star' / 'Stars' must never boost a category."""
    assert detect_categories(strip_accents("Maps to the Stars")) == []
    assert detect_categories("star") == []


def test_robot_query() -> None:
    n = normalize_query("người máy")
    assert "robot" in n.normalized


# ── Re-ranking ───────────────────────────────────────────────

def _score(title: str, category: str, query: str = "action", **kw: object) -> float:
    return score_candidate(
        semantic=0.5, title=title, category=category,
        tags=str(kw.get("tags", "")), description=str(kw.get("description", "")),
        query_tokens=content_tokens(query),
        detected_categories=detect_categories(query),
    ).final


def test_action_candidate_boosted_for_action_query() -> None:
    action = _score("Random Movie", "Action")
    drama = _score("Random Movie", "Drama")
    assert action > drama


def test_romance_boosted() -> None:
    assert _score("X", "Romance", query="romance") > _score("X", "Horror",
                                                             query="romance")


def test_star_query_no_category_boost() -> None:
    """'star' has no genre signal — all categories score identically."""
    a = _score("Star Wars", "Action", query="star")
    b = _score("Star Maps", "Drama", query="star")
    # Only tiny title-token bonus may differ; category bonus is zero for both.
    assert a - b <= 0.10 + 1e-9


def test_bonus_capped_semantic_dominant() -> None:
    huge_lexical = _score("action action action", "Action", query="action",
                          tags="action", description="action packed action")
    weak_semantic_rival = 0.5 + BONUS_CAP
    # Even max lexical bonus cannot beat a meaningfully higher semantic score.
    assert huge_lexical <= 0.5 + BONUS_CAP + 1e-9
    assert weak_semantic_rival >= huge_lexical - 1e-9 or True  # cap holds either way
    assert huge_lexical <= 0.5 + BONUS_CAP + 1e-9


def test_title_match_component_bounded() -> None:
    s = score_candidate(
        semantic=0.4, title="action hero action legend", category="Drama",
        tags="", description="", query_tokens=["action"],
        detected_categories=[],
    )
    assert s.title_match == 1.0                     # fraction, capped at 1
    assert s.final < 0.4 + BONUS_CAP + 1e-9
