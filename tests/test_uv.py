from datetime import date, datetime, time

import pytest

from xp_family.uv import UVSlot, format_phone_e164, is_uv_safe, pick_best_slot

D = date(2026, 7, 1)
SLOTS = [
    UVSlot(D, time(8), 1.0),
    UVSlot(D, time(12), 9.0),
    UVSlot(D, time(16), 2.5),
    UVSlot(D, time(17), None),  # missing forecast
]
EARLY = datetime(2026, 7, 1, 6, 0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("95 445 004", "+21695445004"),
        ("+216 95-445-004", "+21695445004"),
        ("0021695445004", "+21695445004"),
        ("21695445004", "+21695445004"),
        ("+33 6 12 34 56 78", "+33612345678"),
        ("12345", None),
        ("", None),
        ("phone", None),
    ],
)
def test_format_phone_e164(raw, expected):
    assert format_phone_e164(raw) == expected


def test_missing_forecast_is_not_safe():
    assert not is_uv_safe(None)
    assert is_uv_safe(2.9)
    assert not is_uv_safe(3.0)


def test_xp_patient_never_gets_high_or_unknown_uv():
    slot = pick_best_slot(SLOTS, is_xp=True, booked=set(), earliest=EARLY, requested_time=time(12))
    assert slot is not None
    assert slot.uv_index is not None and slot.uv_index < 3


def test_non_xp_patient_gets_closest_to_requested_time():
    slot = pick_best_slot(SLOTS, is_xp=False, booked=set(), earliest=EARLY, requested_time=time(12))
    assert slot is not None and slot.time == time(12)


def test_booked_and_past_slots_are_skipped():
    slot = pick_best_slot(
        SLOTS, is_xp=True, booked={(D, time(16))}, earliest=datetime(2026, 7, 1, 9, 0)
    )
    assert slot is None
