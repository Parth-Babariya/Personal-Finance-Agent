"""Resolve named time periods ("this_month", "last_7_days", ...) to date ranges."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, get_args

Period = Literal[
    "today",
    "this_week",
    "last_7_days",
    "this_month",
    "last_month",
    "last_30_days",
    "this_year",
    "last_year",
    "all_time",
]
PERIODS: tuple[str, ...] = get_args(Period)

# Short aliases accepted by the REST API for backwards compatibility.
_ALIASES = {"day": "today", "week": "last_7_days", "month": "this_month", "year": "this_year", "all": "all_time"}


def normalize_period(period: str) -> str:
    key = period.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in PERIODS:
        raise ValueError(f"Unknown period '{period}'. Use one of: {', '.join(PERIODS)}")
    return key


def resolve_period(period: str, today: date) -> tuple[date | None, date | None]:
    """Return an inclusive (start, end) range; ``(None, None)`` means all time."""
    key = normalize_period(period)

    if key == "today":
        return today, today
    if key == "this_week":
        return today - timedelta(days=today.weekday()), today
    if key == "last_7_days":
        return today - timedelta(days=6), today
    if key == "this_month":
        return today.replace(day=1), today
    if key == "last_month":
        last_day = today.replace(day=1) - timedelta(days=1)
        return last_day.replace(day=1), last_day
    if key == "last_30_days":
        return today - timedelta(days=29), today
    if key == "this_year":
        return today.replace(month=1, day=1), today
    if key == "last_year":
        return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)
    return None, None


def describe_range(start: date | None, end: date | None) -> str:
    if start is None and end is None:
        return "all time"
    if start is None:
        return f"up to {end.isoformat()}"
    if end is None:
        return f"since {start.isoformat()}"
    if start == end:
        return start.isoformat()
    return f"{start.isoformat()} to {end.isoformat()}"
