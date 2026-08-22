"""Metadata-aware re-ranking for semantic search results (query-time only).

Pipeline:  query → semantic retrieval (Qdrant, unchanged) → top-N candidates
           → metadata relevance scoring → final ranking.

Design (conservative by construction):
  - The semantic cosine score stays the dominant term.
  - Lexical bonuses are small, capped, and evidence-weighted:
        category match (strongest) > title phrase match > tags > description
  - Category boost fires ONLY on an explicit genre signal detected in the
    query (see app.query_norm) — a word like "star" never boosts anything.
  - Lexical evidence can therefore reorder near-ties but can never override
    a clear semantic gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.query_norm import strip_accents

# Bonus weights — deliberately small; total bonus is capped at BONUS_CAP.
CATEGORY_BONUS = 0.15
TITLE_TOKEN_BONUS = 0.05   # per matching title token, up to TITLE_TOKEN_CAP
TITLE_TOKEN_CAP = 0.10
TAG_BONUS = 0.03
DESCRIPTION_BONUS = 0.02
BONUS_CAP = 0.25

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "in", "on", "for", "to", "with", "movie",
    "movies", "film", "phim", "about", "some", "like", "want", "give", "me",
})


def content_tokens(text: str) -> list[str]:
    """Accent-stripped lowercase content tokens (stopwords + len<3 removed)."""
    return [
        t for t in _TOKEN_RE.findall(strip_accents(text.lower()))
        if len(t) >= 3 and t not in _STOPWORDS
    ]


@dataclass
class RerankScore:
    semantic: float
    title_match: float
    category_match: float
    tag_match: float
    description_match: float
    final: float


def score_candidate(
    *,
    semantic: float,
    title: str,
    category: str,
    tags: str,
    description: str,
    query_tokens: list[str],
    detected_categories: list[str],
) -> RerankScore:
    """Score one candidate. All components are in [0, 1]; bonus capped."""
    cat_hit = (
        1.0
        if detected_categories
        and any(c.lower() == category.strip().lower() for c in detected_categories)
        else 0.0
    )
    title_l = strip_accents(title.lower())
    title_hits = sum(1 for t in query_tokens if t in title_l)
    tags_l = strip_accents(tags.lower())
    tag_hits = sum(1 for t in query_tokens if t in tags_l)
    desc_l = strip_accents(description.lower())
    desc_hits = sum(1 for t in query_tokens if t in desc_l)

    bonus = (
        CATEGORY_BONUS * cat_hit
        + min(TITLE_TOKEN_CAP, TITLE_TOKEN_BONUS * title_hits)
        + TAG_BONUS * (1.0 if tag_hits else 0.0)
        + DESCRIPTION_BONUS * (1.0 if desc_hits else 0.0)
    )
    bonus = min(BONUS_CAP, bonus)
    return RerankScore(
        semantic=semantic,
        title_match=min(1.0, title_hits / max(1, len(query_tokens))),
        category_match=cat_hit,
        tag_match=1.0 if tag_hits else 0.0,
        description_match=1.0 if desc_hits else 0.0,
        final=semantic + bonus,
    )
