"""Record-aware chunking.

The original pipeline embedded whole records (assistants 1-2) or split recipe text on a
fixed 512-character window (assistant 3), which cut ingredient lists mid-name and left
chunks with no indication of which product they came from. Here:

* short records stay whole, so one substance = one chunk;
* long records are split on line, then sentence, boundaries — never mid-word;
* every chunk is prefixed with its record title, so a chunk retrieved on its own
  still says what it is about (contextual chunk headers).
"""

from __future__ import annotations

import re
from dataclasses import replace

from xp_family.rag.documents import Document

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def _pieces(text: str) -> list[str]:
    """Split into the smallest natural units: lines, then sentences within long lines."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.extend(p for p in _SENTENCE_END.split(line) if p)
    return out


def _hard_wrap(piece: str, size: int) -> list[str]:
    """Last resort for a single unit longer than `size`: wrap on whitespace."""
    words, lines, current = piece.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > size:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def chunk_document(doc: Document, size: int = 800, overlap: int = 120) -> list[Document]:
    header = f"[{doc.title}]"
    budget = max(size - len(header) - 1, 50)
    if len(doc.text) <= budget:
        return [replace(doc, id=f"{doc.id}#0", text=f"{header}\n{doc.text}")]

    units: list[str] = []
    for piece in _pieces(doc.text):
        units.extend(_hard_wrap(piece, budget) if len(piece) > budget else [piece])

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and len("\n".join([*current, unit])) > budget:
            chunks.append("\n".join(current))
            # Carry trailing units forward as overlap, without exceeding the budget.
            carried: list[str] = []
            for prev in reversed(current):
                if len("\n".join([prev, *carried])) > overlap:
                    break
                carried.insert(0, prev)
            if len("\n".join([*carried, unit])) > budget:
                carried = []
            current = carried
        current.append(unit)
    if current:
        chunks.append("\n".join(current))

    return [
        replace(doc, id=f"{doc.id}#{i}", text=f"{header}\n{body}") for i, body in enumerate(chunks)
    ]


def chunk_documents(docs: list[Document], size: int = 800, overlap: int = 120) -> list[Document]:
    return [chunk for doc in docs for chunk in chunk_document(doc, size, overlap)]
