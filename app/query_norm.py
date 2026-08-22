"""Query normalization for Vietnamese movie-domain queries (query-time only).

Scope is deliberately narrow: a curated dictionary of movie-domain phrases,
NOT free translation. The embedding model (all-MiniLM-L6-v2) is English-only,
so Vietnamese queries are mapped to their English equivalents before encoding.
The Qdrant collection and the model are unchanged — no re-index required.

Guarantees:
  - Original query is preserved for the API response / UI.
  - Accent-insensitive matching: "PHIM HÀNH ĐỘNG", "Phim Hành Động" and
    "phim hành động" all normalize identically (Unicode NFD strip, never ASCII
    drop of meaning).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

# Vietnamese function/filler words — carry no movie intent and poison the
# English embedding ("toi muon xem horror movie" → junk). Stripped from the
# normalized query before encoding.
_VI_FILLERS: frozenset[str] = frozenset({
    "toi", "muon", "xem", "thich", "tim", "can", "mot", "mot", "nhung",
    "nao", "gi", "vay", "cho", "voi", "la", "cua", "va", "hoac", "goi_y",
    "bo", "dien", "hay",
})

# Accent-stripped Vietnamese phrase → English replacement (longest-first applied).
_VI_PHRASES: dict[str, str] = {
    "phim khoa hoc vien tuong": "science fiction movie",
    "phime khoa hoc vien tuong": "science fiction movie",
    "phim sieu anh hung": "superhero movie",
    "phim hanh dong": "action movie",
    "phim tinh cam": "romance movie",
    "phim kinh di": "horror movie",
    "phim chien tranh": "war movie",
    "phim hoat hinh": "animation movie",
    "phim tre em": "children movie",
    "phim vu tru": "space movie",
    "phim my": "american movie",
    "phim hai": "comedy movie",
    "nguoi may": "robot",
    "vu tru": "space",
}

# Single-word fallbacks after phrase replacement.
_VI_WORDS: dict[str, str] = {
    "phim": "movie",
    "hanh": "action",
    "dong": "action",
    "tinh": "romance",
    "cam": "romance",
    "hai": "comedy",
    "kinh": "horror",
    "di": "horror",
    "vien": "fiction",
    "tuong": "fiction",
    "khoa": "science",
    "hoc": "science",
    "chien": "war",
    "tranh": "war",
    "hoat": "animation",
    "hinh": "animation",
    "tre": "children",
    "em": "children",
    "sieu": "super",
    "hung": "hero",
    "nguoi": "robot",
    "may": "robot",
    "my": "american",
}

# Accent-stripped query phrase → canonical MovieLens category label.
# Exact phrase match ONLY — a generic word like "star" must never map to a
# category just because some title contains it (false-positive guard).
_GENRE_ALIASES: dict[str, str] = {
    "science fiction": "Sci-Fi",
    "sci fi": "Sci-Fi",
    "khoa hoc vien tuong": "Sci-Fi",
    "superhero": "Action",
    "sieu anh hung": "Action",
    "action": "Action",
    "hanh dong": "Action",
    "comedy": "Comedy",
    "hai": "Comedy",
    "romance": "Romance",
    "tinh cam": "Romance",
    "horror": "Horror",
    "kinh di": "Horror",
    "adventure": "Adventure",
    "phuu luu": "Adventure",
    "animation": "Animation",
    "hoat hinh": "Animation",
    "children": "Children",
    "tre em": "Children",
    "war": "War",
    "chien tranh": "War",
    "crime": "Crime",
    "toi pham": "Crime",
    "drama": "Drama",
    "chinh kich": "Drama",
    "fantasy": "Fantasy",
    "thriller": "Thriller",
    "western": "Western",
    "musical": "Musical",
    "mystery": "Mystery",
    "documentary": "Documentary",
    "film noir": "Film-Noir",
}

# Sorted longest-first so "khoa hoc vien tuong" wins over its sub-tokens.
_PHRASE_ORDER = sorted(_VI_PHRASES, key=len, reverse=True)
_GENRE_ORDER = sorted(_GENRE_ALIASES, key=len, reverse=True)


def strip_accents(text: str) -> str:
    """Accent-insensitive key: Unicode NFD + drop combining marks (never drops letters).
    'đ'/'Đ' (non-decomposable Vietnamese letters) are mapped to 'd' explicitly."""
    replaced = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", replaced.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@dataclass
class NormalizedQuery:
    original: str
    normalized: str  # English equivalent used for embedding
    categories: list[str] = field(default_factory=list)


def detect_categories(key: str) -> list[str]:
    """Detect known movie categories from an accent-stripped query key."""
    padded = f" {key} "
    found: list[str] = []
    for phrase in _GENRE_ORDER:
        label = _GENRE_ALIASES[phrase]
        if label in found:
            continue
        if f" {phrase} " in padded:
            found.append(label)
            padded = padded.replace(f" {phrase} ", " ")
    return found


def normalize_query(query: str) -> NormalizedQuery:
    """Map a (possibly Vietnamese) movie query onto its English equivalent."""
    key = strip_accents(query)
    out = key
    for phrase in _PHRASE_ORDER:
        if phrase in out:
            out = out.replace(phrase, _VI_PHRASES[phrase])
    # Word-level pass on remaining Vietnamese tokens; fillers are dropped.
    words = []
    for w in out.split():
        if w in _VI_FILLERS:
            continue
        if w in _VI_WORDS or not w.isascii():
            words.append(_VI_WORDS.get(w, w))
        else:
            words.append(w)
    normalized = " ".join(words).strip() or key
    return NormalizedQuery(
        original=query, normalized=normalized, categories=detect_categories(key)
    )
