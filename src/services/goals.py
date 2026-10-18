"""Savings goals with progress tracking against the user's actual saving rate."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from ..interfaces.transaction_service import ITransactionService
from .models import SavingsGoal
from .sql_transaction_service import to_decimal

SAVING_RATE_MONTHS = 3  # complete months used to estimate the monthly saving rate


def _shift_month(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + end.month - start.month


class GoalService:
    def __init__(
        self,
        session_factory: sessionmaker,
        tx_service: ITransactionService,
        today: Callable[[], date] = date.today,
    ):
        self._session_factory = session_factory
        self._tx = tx_service
        self._today = today

    def create(
        self, user_id: str, name: str, target_amount: float, target_date: date | None = None, saved_amount: float = 0
    ) -> dict[str, Any]:
        if target_date is not None and target_date <= self._today():
            raise ValueError("target_date must be in the future")
        with self._session_factory() as session:
            row = SavingsGoal(
                user_id=user_id, name=name, target_amount=to_decimal(target_amount),
                saved_amount=to_decimal(saved_amount), target_date=target_date,
            )
            session.add(row)
            session.commit()
            return self._with_progress(row, self.average_monthly_saving(user_id))

    def list(self, user_id: str) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(SavingsGoal).where(SavingsGoal.user_id == user_id).order_by(SavingsGoal.created_at)
            ).all()
        rate = self.average_monthly_saving(user_id) if rows else 0.0
        return [self._with_progress(row, rate) for row in rows]

    def contribute(self, goal_id: str, user_id: str, amount: float) -> dict[str, Any] | None:
        with self._session_factory() as session:
            row = self._row(session, goal_id, user_id)
            if row is None:
                return None
            row.saved_amount = to_decimal(row.saved_amount + to_decimal(amount))
            if row.saved_amount < 0:
                raise ValueError("Saved amount cannot go below zero")
            session.commit()
            return self._with_progress(row, self.average_monthly_saving(user_id))

    def delete(self, goal_id: str, user_id: str) -> bool:
        with self._session_factory() as session:
            row = self._row(session, goal_id, user_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def average_monthly_saving(self, user_id: str) -> float:
        """Average income minus expenses over the last few complete months."""
        this_month = self._today().replace(day=1)
        nets = []
        for i in range(1, SAVING_RATE_MONTHS + 1):
            start = _shift_month(this_month, -i)
            end = _shift_month(start, 1).replace(day=1)
            summary = self._tx.get_summary(user_id, start=start, end=date.fromordinal(end.toordinal() - 1))
            if summary["transaction_count"]:
                nets.append(summary["net"])
        return round(sum(nets) / len(nets), 2) if nets else 0.0

    @staticmethod
    def _row(session, goal_id: str, user_id: str) -> SavingsGoal | None:
        return session.scalar(select(SavingsGoal).where(SavingsGoal.id == goal_id, SavingsGoal.user_id == user_id))

    def _with_progress(self, row: SavingsGoal, monthly_saving: float) -> dict[str, Any]:
        today = self._today()
        target, saved = float(row.target_amount), float(row.saved_amount)
        remaining = round(max(target - saved, 0.0), 2)
        goal: dict[str, Any] = {
            "id": row.id,
            "name": row.name,
            "target_amount": target,
            "saved_amount": saved,
            "remaining": remaining,
            "progress_pct": round(min(saved / target, 1.0) * 100, 1) if target else 100.0,
            "completed": remaining == 0,
            "target_date": row.target_date.isoformat() if row.target_date else None,
            "average_monthly_saving": monthly_saving,
            "required_per_month": None,
            "on_track": None,
            "projected_completion": None,
        }
        if goal["completed"]:
            return goal
        if row.target_date:
            months_left = max(_months_between(today, row.target_date), 1)
            goal["months_left"] = months_left
            goal["required_per_month"] = round(remaining / months_left, 2)
            goal["on_track"] = monthly_saving >= goal["required_per_month"]
        if monthly_saving > 0:
            goal["projected_completion"] = _shift_month(today, math.ceil(remaining / monthly_saving)).strftime("%Y-%m")
        return goal
