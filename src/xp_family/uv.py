"""UV-aware appointment slot selection — pure logic, extracted from the Django view.

XP patients must never be booked into a slot whose forecast UV index is at or above
the safety threshold; other patients may be, but the booking records it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time

SAFE_UV_INDEX = 3.0
_SEPARATORS = re.compile(r"[\s\-().]+")


@dataclass(frozen=True)
class UVSlot:
    date: date
    time: time
    uv_index: float | None  # None when the forecast is missing

    @property
    def start(self) -> datetime:
        return datetime.combine(self.date, self.time)


def is_uv_safe(uv_index: float | None, threshold: float = SAFE_UV_INDEX) -> bool:
    """Missing forecasts are treated as unsafe — never assume safety for XP."""
    return uv_index is not None and uv_index < threshold


def format_phone_e164(raw: str | None, default_country: str = "216") -> str | None:
    """Normalise a phone number to E.164. Tunisian 8-digit local numbers get +216.

    Returns None when the number cannot be interpreted, so callers skip the SMS
    rather than texting a wrong number.
    """
    if not raw:
        return None
    phone = _SEPARATORS.sub("", raw.strip())
    if phone.startswith("00"):
        phone = "+" + phone[2:]
    if phone.startswith("+"):
        digits = phone[1:]
        return phone if digits.isdigit() and 8 <= len(digits) <= 15 else None
    if not phone.isdigit():
        return None
    if len(phone) == 8:
        return f"+{default_country}{phone}"
    if phone.startswith(default_country) and len(phone) == len(default_country) + 8:
        return f"+{phone}"
    return None


def pick_best_slot(
    slots: Iterable[UVSlot],
    *,
    is_xp: bool,
    booked: set[tuple[date, time]],
    earliest: datetime,
    work_start: time = time(7, 0),
    work_end: time = time(18, 0),
    requested_date: date | None = None,
    requested_time: time | None = None,
    threshold: float = SAFE_UV_INDEX,
) -> UVSlot | None:
    """Choose a free slot: UV-safe only for XP patients, closest to the request if any."""
    candidates = [
        s
        for s in slots
        if s.start >= earliest
        and work_start <= s.time <= work_end
        and (s.date, s.time) not in booked
        and (not is_xp or is_uv_safe(s.uv_index, threshold))
        and (requested_date is None or s.date == requested_date)
    ]
    if not candidates:
        return None
    if requested_time is not None:
        wanted = requested_time.hour * 60 + requested_time.minute

        def distance(s: UVSlot) -> tuple[int, date, time]:
            return (abs(s.time.hour * 60 + s.time.minute - wanted), s.date, s.time)

        return min(candidates, key=distance)
    return min(candidates, key=lambda s: (s.date, s.time))
