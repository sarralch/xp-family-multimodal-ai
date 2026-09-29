"""Deterministic XP ingredient screening.

v1 treated any ingredient absent from the banned list as safe. For a tool used by
families of children with XP that is the wrong default, so every ingredient now gets
one of four explicit statuses and "unknown" is surfaced, never silently passed.
Matching is accent- and case-insensitive and covers names, synonyms and CAS numbers.

The engine never declares an ingredient *safe*. Being in the ingredient-information
file without an XP warning only means "no warning in our data" (NOT_FLAGGED): in the
project dataset every entry carries safe_for_xp=true, including photosensitising
retinol, so that flag cannot support a safety claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from xp_family.rag.documents import load_json
from xp_family.rag.text import fold

_CAS = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
_SPLIT = re.compile(r"[,;\n•·]+")
_PARENS = re.compile(r"\([^)]*\)")
_LABEL = re.compile(r"^\s*(ingr[eé]dients?|ingredients?|inci)\s*[:：]\s*", re.IGNORECASE)


class Status(StrEnum):
    BANNED = "banned"  # on the regulatory / XP banned list
    CAUTION = "caution"  # known ingredient, explicitly flagged as not suitable for XP
    NOT_FLAGGED = "not_flagged"  # known ingredient, no XP warning in our data (not "safe")
    UNKNOWN = "unknown"  # not in any list — needs professional review


@dataclass(frozen=True)
class Verdict:
    ingredient: str
    status: Status
    matched: str | None = None
    effects: list[str] = field(default_factory=list)


def _key(name: str) -> str:
    return " ".join(fold(name).split())


def parse_ingredient_list(text: str) -> list[str]:
    """Split an OCR'd or typed INCI list into individual ingredient names."""
    text = _LABEL.sub("", text.strip())
    items = (" ".join(part.split()).strip(" .:") for part in _SPLIT.split(text))
    seen: set[str] = set()
    out = []
    for item in items:
        if item and _key(item) not in seen:
            seen.add(_key(item))
            out.append(item)
    return out


class SafetyChecker:
    def __init__(
        self,
        banned: dict[str, dict[str, Any]],
        ingredient_info: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._banned: dict[str, tuple[str, list[str]]] = {}
        for key, entry in banned.items():
            label = entry.get("name") or key
            effects = [e for e in entry.get("effects") or [] if e]
            names = [key, label, *(entry.get("synonyms") or [])]
            names += _CAS.findall(entry.get("cas") or "")
            for name in names:
                if name:
                    self._banned[_key(name)] = (label, effects)

        self._info: dict[str, tuple[str, bool]] = {}
        for key, entry in (ingredient_info or {}).items():
            if _key(key) in ("", "nan", "none", "null"):  # export artefacts
                continue
            # Only an explicit `false` is a warning; a missing flag is not.
            self._info[_key(key)] = (key, entry.get("safe_for_xp") is not False)

    @classmethod
    def from_files(cls, banned_path: Path, info_path: Path | None = None) -> SafetyChecker:
        info = load_json(info_path) if info_path and info_path.exists() else None
        return cls(load_json(banned_path), info)

    def _candidates(self, ingredient: str) -> list[str]:
        """'Aqua (Water)' -> ['aqua (water)', 'aqua', 'water'], plus any CAS numbers."""
        full = _key(ingredient)
        outside = _key(_PARENS.sub(" ", ingredient))
        inside = [_key(m) for m in re.findall(r"\(([^)]*)\)", ingredient)]
        slashed = [_key(p) for p in re.split(r"/", outside) if p.strip()]
        return list(dict.fromkeys([full, outside, *inside, *slashed, *_CAS.findall(ingredient)]))

    def check_one(self, ingredient: str) -> Verdict:
        candidates = self._candidates(ingredient)
        for cand in candidates:
            if cand in self._banned:
                label, effects = self._banned[cand]
                return Verdict(ingredient, Status.BANNED, label, effects)
        for cand in candidates:
            if cand in self._info:
                label, unflagged = self._info[cand]
                status = Status.NOT_FLAGGED if unflagged else Status.CAUTION
                return Verdict(ingredient, status, label)
        return Verdict(ingredient, Status.UNKNOWN)

    def check(self, ingredients: list[str]) -> list[Verdict]:
        return [self.check_one(i) for i in ingredients]


def summarize(verdicts: list[Verdict]) -> dict[str, int]:
    return {status.value: sum(v.status is status for v in verdicts) for status in Status}
