"""Text normalisation shared by BM25, the hashing embedder and the safety checker."""

from __future__ import annotations

import re
import unicodedata

# Keep hyphenated chemical names and CAS numbers (e.g. 106-88-7) as single tokens.
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def fold(text: str) -> str:
    """Lowercase and strip accents: 'Époxybutane' -> 'epoxybutane'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(fold(text))
