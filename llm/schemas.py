"""Structured LLM output schemas (LLMOps 10A.3).

The LLM never outputs freeform text for machine-readable steps; it is constrained
to these Pydantic models. Any hallucinated id or unknown field is rejected by the
validator in :mod:`llm.service`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal[
    "jacket", "coat", "shirt", "t-shirt", "sweater", "hoodie", "dress",
    "jeans", "pants", "shorts", "skirt", "shoes", "sneakers", "boots",
    "hat", "scarf", "accessories", "other",
]


class PreferenceExtraction(BaseModel):
    """Structured preference extracted from a natural-language request (10A.1-A).

    Example:
        {"category": "jacket", "color": "black", "style": "casual",
         "price_preference": "medium_or_lower"}
    """

    category: Category | None = None
    color: str | None = None
    style: str | None = None
    price_preference: Literal["low", "medium", "high", "medium_or_lower"] | None = None

    @field_validator("color", "style")
    @classmethod
    def _strip(cls, v):  # noqa: ANN001, ANN202
        if isinstance(v, str):
            v = v.strip().lower()
            return v or None
        return v


class StructuredFilter(BaseModel):
    """Structured filter used to build the vector/retrieval query (LOMM 10A.2)."""

    category: str | None = None
    color: str | None = None
    styles: list[str] = Field(default_factory=list)


class Explanation(BaseModel):
    """Natural-language explanation grounded in real item data (never hallucinated)."""

    text: str
    item_attributes: list[str] = Field(default_factory=list)


class ItemPrediction(BaseModel):
    """One recommended item in an LLM-generated chat response.

    item_id MUST come from the real retrieval/ranking result set; any id not in
    the supplied catalog is rejected before the response is returned.
    """

    item_id: str
    reason: str = ""


# Alias kept for back-compat with llm/__init__ imports that referenced ItemFilter.
ItemFilter = PreferenceExtraction
