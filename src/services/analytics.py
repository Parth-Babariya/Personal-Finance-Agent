"""Spending analytics: monthly trends, unusual expenses and recurring payments.

Computed in Python over the user's transactions, which is fine for personal
data volumes (thousands of rows). Rules are deliberately simple and explainable
so every flag comes with a human-readable reason.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from src.interfaces.transaction_service import ITransactionService

ANOMALY_BASELINE_DAYS = 180
MIN_CATEGORY_SAMPLES = 5
CATEGORY_MEDIAN_MULTIPLE = 2.5
CATEGORY_STDEV_MULTIPLE = 2.5
OVERALL_MEDIAN_MULTIPLE = 10

RECURRING_MIN_MONTHS = 3
RECURRING_MAX_AMOUNT_SPREAD = 0.3  # max deviation from the median amount
RECURRING_MAX_PER_MONTH = 1.5


def _month_key(day: date) -> str:
    return day.strftime("%Y-%m")


def _shift_month(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def payee_key(description: str) -> str:
    """Normalize a description so 'Netflix subscription #123' groups with 'NETFLIX Subscription'."""
    text = re.sub(r"[^a-z ]", " ", description.lower())
    return " ".join(w for w in text.split() if len(w) > 1)


class AnalyticsService:
    def __init__(self, tx_service: ITransactionService, today: Callable[[], date] = date.today):
        self._tx = tx_service
        self._today = today

    def today(self) -> date:
        return self._today()

    def monthly_trend(self, user_id: str, months: int = 6, category: str | None = None) -> dict[str, Any]:
        today = self._today()
        first = _shift_month(today, -(months - 1))
        rows = self._tx.list_transactions(user_id, limit=0, start=first, end=today, category=category)
        buckets: dict[str, dict[str, float]] = {
            _month_key(_shift_month(first, i)): {"expense": 0.0, "income": 0.0} for i in range(months)
        }
        for row in rows:
            buckets[row["date"][:7]][row["type"]] += row["amount"]
        series = [
            {"month": month, "expense": round(v["expense"], 2), "income": round(v["income"], 2),
             "net": round(v["income"] - v["expense"], 2), "partial": month == _month_key(today)}
            for month, v in buckets.items()
        ]
        complete = [m["expense"] for m in series if not m["partial"]]
        return {
            "category": category,
            "months": series,
            "average_monthly_expense": round(statistics.mean(complete), 2) if complete else None,
        }

    def unusual_expenses(self, user_id: str, start: date | None, end: date | None) -> list[dict[str, Any]]:
        today = self._today()
        window_start, window_end = start or today - timedelta(days=30), end or today
        baseline_rows = self._tx.list_transactions(
            user_id, limit=0, type="expense",
            start=min(window_start, today - timedelta(days=ANOMALY_BASELINE_DAYS)), end=window_end,
        )
        recurring = {r["payee"] for r in self._recurring_groups(baseline_rows)}
        overall_median = statistics.median([r["amount"] for r in baseline_rows]) if baseline_rows else 0.0
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in baseline_rows:
            by_category[row["category"]].append(row)

        flagged = []
        for row in baseline_rows:
            when = date.fromisoformat(row["date"])
            if not (window_start <= when <= window_end) or payee_key(row["description"]) in recurring:
                continue
            others = [r["amount"] for r in by_category[row["category"]] if r["id"] != row["id"]]
            reason = None
            if len(others) >= MIN_CATEGORY_SAMPLES:
                median, stdev = statistics.median(others), statistics.pstdev(others)
                far_above = row["amount"] > median + CATEGORY_STDEV_MULTIPLE * stdev
                if row["amount"] > CATEGORY_MEDIAN_MULTIPLE * median and far_above:
                    reason = (f"{row['amount'] / median:.1f}x your typical {row['category']} expense "
                              f"(median {median:,.2f})")
            if reason is None and overall_median and row["amount"] > OVERALL_MEDIAN_MULTIPLE * overall_median:
                reason = f"{row['amount'] / overall_median:.0f}x your typical expense (median {overall_median:,.2f})"
            if reason:
                flagged.append({**row, "reason": reason})
        return sorted(flagged, key=lambda r: r["amount"], reverse=True)

    def recurring_payments(self, user_id: str) -> dict[str, Any]:
        today = self._today()
        rows = self._tx.list_transactions(user_id, limit=0, type="expense", start=today - timedelta(days=365))
        groups = self._recurring_groups(rows)
        return {
            "recurring": groups,
            "estimated_monthly_total": round(sum(g["typical_amount"] for g in groups), 2),
        }

    def _recurring_groups(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_payee: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = payee_key(row["description"])
            if key:
                by_payee[key].append(row)

        groups = []
        for key, items in by_payee.items():
            months = {r["date"][:7] for r in items}
            if len(months) < RECURRING_MIN_MONTHS or len(items) / len(months) > RECURRING_MAX_PER_MONTH:
                continue
            amounts = [r["amount"] for r in items]
            median = statistics.median(amounts)
            spread = max(abs(a - median) for a in amounts) / median if median else 1.0
            if spread > RECURRING_MAX_AMOUNT_SPREAD:
                continue
            last = max(items, key=lambda r: r["date"])
            last_date = date.fromisoformat(last["date"])
            groups.append({
                "payee": key,
                "description": last["description"],
                "category": last["category"],
                "typical_amount": round(median, 2),
                "fixed_amount": spread < 0.01,
                "months_seen": len(months),
                "last_date": last["date"],
                "next_expected": (last_date + timedelta(days=30)).isoformat(),
            })
        return sorted(groups, key=lambda g: g["typical_amount"], reverse=True)
