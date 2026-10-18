"""Monthly category budgets with usage status and alerts."""

from __future__ import annotations

import calendar
from collections.abc import Callable
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from src.db.models import Budget
from src.interfaces.transaction_service import ITransactionService
from src.services.transactions import to_decimal

WARN_AT = 0.8  # warn when 80% of a budget is used


def _month_bounds(day: date) -> tuple[date, date]:
    return day.replace(day=1), day.replace(day=calendar.monthrange(day.year, day.month)[1])


def _state(pct: float) -> str:
    if pct >= 1:
        return "over"
    return "warning" if pct >= WARN_AT else "ok"


class BudgetService:
    def __init__(
        self,
        session_factory: sessionmaker,
        tx_service: ITransactionService,
        today: Callable[[], date] = date.today,
        currency: str = "₹",
    ):
        self._session_factory = session_factory
        self._tx = tx_service
        self._today = today
        self._cur = currency

    def today(self) -> date:
        return self._today()

    # --- CRUD ---

    def set_budget(self, user_id: str, category: str, monthly_limit: float) -> dict[str, Any]:
        with self._session_factory() as session:
            row = session.scalar(select(Budget).where(Budget.user_id == user_id, Budget.category == category))
            if row is None:
                row = Budget(user_id=user_id, category=category, monthly_limit=to_decimal(monthly_limit))
                session.add(row)
            else:
                row.monthly_limit = to_decimal(monthly_limit)
            session.commit()
            return {"category": row.category, "monthly_limit": float(row.monthly_limit)}

    def delete_budget(self, user_id: str, category: str) -> bool:
        with self._session_factory() as session:
            row = session.scalar(select(Budget).where(Budget.user_id == user_id, Budget.category == category))
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def list_budgets(self, user_id: str) -> dict[str, float]:
        with self._session_factory() as session:
            rows = session.scalars(select(Budget).where(Budget.user_id == user_id).order_by(Budget.category))
            return {row.category: float(row.monthly_limit) for row in rows}

    # --- status ---

    def status(self, user_id: str, category: str | None = None, month: date | None = None) -> list[dict[str, Any]]:
        """Usage of each budget for the month containing ``month`` (default: current month)."""
        today = self._today()
        start, month_end = _month_bounds(month or today)
        is_current = start <= today <= month_end
        end = today if is_current else month_end
        days_in_month = month_end.day
        out = []
        for cat, limit in self.list_budgets(user_id).items():
            if category and cat != category:
                continue
            spent = self._tx.get_summary(user_id, start=start, end=end, category=cat)["total_expense"]
            pct = spent / limit if limit else 0.0
            entry = {
                "category": cat,
                "month": start.strftime("%Y-%m"),
                "limit": limit,
                "spent": spent,
                "remaining": round(limit - spent, 2),
                "used_pct": round(pct * 100, 1),
                "state": _state(pct),
            }
            if is_current:
                projected = spent / today.day * days_in_month
                entry["projected_month_end"] = round(projected, 2)
                entry["projected_state"] = _state(projected / limit if limit else 0.0)
            out.append(entry)
        return out

    def impact(self, user_id: str, category: str, amount: float, when: date) -> dict[str, Any] | None:
        """How a prospective expense would change its category budget (None if no budget applies)."""
        limit = self.list_budgets(user_id).get(category)
        if limit is None:
            return None
        start, end = _month_bounds(when)
        spent = self._tx.get_summary(user_id, start=start, end=end, category=category)["total_expense"]
        after = spent + amount
        return {
            "category": category,
            "limit": limit,
            "spent_before": spent,
            "spent_after": round(after, 2),
            "used_pct_after": round(after / limit * 100, 1) if limit else 0.0,
            "state_after": _state(after / limit if limit else 0.0),
        }

    def alert_for(self, user_id: str, category: str, when: date) -> str | None:
        """A human-readable warning if ``category`` is at or above the warning threshold."""
        [entry] = self.status(user_id, category, month=when) or [None]
        if entry is None or entry["state"] == "ok":
            return None
        used = f"{self._cur}{entry['spent']:,.2f} of {self._cur}{entry['limit']:,.2f}"
        if entry["state"] == "over":
            return f"You're over your {category} budget for {entry['month']} ({used})."
        return f"You've used {entry['used_pct']:.0f}% of your {category} budget for {entry['month']} ({used})."
