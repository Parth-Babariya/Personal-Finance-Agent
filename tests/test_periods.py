from datetime import date

import pytest

from src.services.periods import describe_range, normalize_period, resolve_period

TODAY = date(2026, 3, 18)  # Wednesday


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ("today", (date(2026, 3, 18), date(2026, 3, 18))),
        ("this_week", (date(2026, 3, 16), date(2026, 3, 18))),
        ("last_7_days", (date(2026, 3, 12), date(2026, 3, 18))),
        ("this_month", (date(2026, 3, 1), date(2026, 3, 18))),
        ("last_month", (date(2026, 2, 1), date(2026, 2, 28))),
        ("last_30_days", (date(2026, 2, 17), date(2026, 3, 18))),
        ("this_year", (date(2026, 1, 1), date(2026, 3, 18))),
        ("last_year", (date(2025, 1, 1), date(2025, 12, 31))),
        ("all_time", (None, None)),
    ],
)
def test_resolve_period(period, expected):
    assert resolve_period(period, TODAY) == expected


def test_last_month_across_year_boundary():
    assert resolve_period("last_month", date(2026, 1, 10)) == (date(2025, 12, 1), date(2025, 12, 31))


def test_aliases_and_unknown_period():
    assert normalize_period("Month") == "this_month"
    assert normalize_period("week") == "last_7_days"
    with pytest.raises(ValueError):
        normalize_period("fortnight")


def test_describe_range():
    assert describe_range(None, None) == "all time"
    assert describe_range(date(2026, 3, 1), date(2026, 3, 1)) == "2026-03-01"
    assert describe_range(date(2026, 3, 1), date(2026, 3, 18)) == "2026-03-01 to 2026-03-18"
