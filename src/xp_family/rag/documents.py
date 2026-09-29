"""Document model and loaders for the XP knowledge sources.

The knowledge files are heterogeneous (regulatory tables exported with several key
spellings, plus summaries of research papers), so records are rendered generically:
a title is picked from the first name-like key, and every non-empty field becomes a
`Key: value` line. See `data/README.md` for the expected files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Keys that name the record, in priority order, across the known export variants.
_TITLE_KEYS = (
    "name",
    "nom",
    "titre",
    "substance_chimique",
    "Substance_chimique",
    "Substance chimique",
    "Substance",
    "product_name",
)
_EMPTY = {"", "none", "null", "nan", "[]", "{}"}


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


def _is_empty(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, str | list | dict) and not value)
        or (isinstance(value, str) and value.strip().lower() in _EMPTY)
    )


def _render_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if not _is_empty(v))
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items() if not _is_empty(v))
    return str(value).strip()


def _humanize(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


def record_to_document(record: dict[str, Any], source: str, key: str | None = None) -> Document:
    title = key or next(
        (str(record[k]) for k in _TITLE_KEYS if k in record and not _is_empty(record[k])),
        "Sans titre",
    )
    lines = [
        f"{_humanize(k)}: {_render_value(v)}"
        for k, v in record.items()
        if not _is_empty(v) and _render_value(v)
    ]
    text = "\n".join(lines)
    digest = hashlib.sha1(f"{source}|{title}|{text}".encode()).hexdigest()[:12]
    return Document(id=f"{source}:{digest}", title=title, text=text, source=source, metadata=record)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_records(path: Path, source: str) -> list[Document]:
    """Load a JSON list of records, or a JSON object keyed by record name."""
    data = load_json(path)
    if isinstance(data, dict):
        return [
            record_to_document(
                {"name": k, **v} if isinstance(v, dict) else {"name": k, "value": v}, source
            )
            for k, v in data.items()
        ]
    return [record_to_document(r, source) for r in data if isinstance(r, dict)]


def load_products(path: Path) -> list[Document]:
    """Products annotated with XP safety (`safe_for_xp`, flagged `contains`)."""
    docs = []
    for record in load_json(path):
        ingredients = record.get("ingredients") or []
        if isinstance(ingredients, str):  # tolerate the raw product export
            ingredients = [i.strip() for i in ingredients.split(",") if i.strip()]
        flagged = record.get("contains") or []
        text = "\n".join(
            filter(
                None,
                [
                    f"Produit: {record.get('name') or record.get('product_name', '')}",
                    f"Type: {record.get('type') or record.get('product_type', '')}",
                    f"Ingrédients: {', '.join(ingredients)}",
                    (
                        f"Compatible XP: {'oui' if record['safe_for_xp'] else 'non'}"
                        if "safe_for_xp" in record
                        else ""
                    ),
                    f"Ingrédients signalés: {', '.join(flagged)}" if flagged else "",
                ],
            )
        )
        title = record.get("name") or record.get("product_name") or "Produit"
        digest = hashlib.sha1(f"products|{title}|{text}".encode()).hexdigest()[:12]
        docs.append(
            Document(
                id=f"products:{digest}", title=title, text=text, source="products", metadata=record
            )
        )
    return docs
