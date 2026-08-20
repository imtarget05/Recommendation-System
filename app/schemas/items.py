"""Internal normalized item schema (Section 5.4)."""

from __future__ import annotations

from pydantic import BaseModel

ITEM_COLUMNS = ["item_id", "title", "category", "tags", "text_description"]
ITEM_DTYPES = {
    "item_id": "string",
    "title": "string",
    "category": "string",
    "tags": "string",
    "text_description": "string",
}


class Item(BaseModel):
    item_id: str
    title: str | None = None
    category: str | None = None
    tags: str | None = None
    text_description: str | None = None


def build_item_text(title=None, category=None, tags=None):  # noqa: ANN001
    """Build semantic text from item metadata (Section 7.2)."""
    parts = [p for p in (title, category, tags) if p]
    return " ".join(str(p).strip() for p in parts if str(p).strip())
