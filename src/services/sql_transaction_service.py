"""SQLAlchemy-backed transaction service with user-level isolation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from sqlalchemy import Select, case, func, select
from sqlalchemy.orm import sessionmaker

from ..interfaces.transaction_service import ITransactionService
from .models import Transaction

_UPDATABLE_FIELDS = {"amount", "type", "category", "description", "date"}


def to_decimal(value: Any) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _money(value: Decimal | float | None) -> float:
    return float(to_decimal(value or 0))


def _to_dict(row: Transaction) -> dict[str, Any]:
    return {
        "id": row.id,
        "user_id": row.user_id,
        "type": row.type,
        "amount": _money(row.amount),
        "category": row.category,
        "description": row.description,
        "date": row.date.isoformat(),
    }


class SQLTransactionService(ITransactionService):
    """Persistent transaction service using a relational database."""

    def __init__(self, session_factory: sessionmaker, today: Callable[[], date] = date.today):
        self._session_factory = session_factory
        self._today = today

    def today(self) -> date:
        """Current date per the injected clock (fixed in tests and evals)."""
        return self._today()

    # --- writes ---

    def add_transaction(self, data: dict[str, Any], user_id: str) -> dict[str, Any]:
        record = Transaction(
            user_id=user_id,
            type=data.get("type", "expense"),
            amount=to_decimal(data["amount"]),
            category=data.get("category") or "other",
            description=data.get("description") or "",
            date=data.get("date") or self._today(),
        )
        with self._session_factory() as session:
            session.add(record)
            session.commit()
            return _to_dict(record)

    def update_transaction(
        self, transaction_id: str, changes: dict[str, Any], user_id: str
    ) -> dict[str, Any] | None:
        with self._session_factory() as session:
            row = self._get_row(session, transaction_id, user_id)
            if row is None:
                return None
            for field, value in changes.items():
                if field not in _UPDATABLE_FIELDS or value is None:
                    continue
                setattr(row, field, to_decimal(value) if field == "amount" else value)
            session.commit()
            return _to_dict(row)

    def delete_transaction(self, transaction_id: str, user_id: str) -> bool:
        with self._session_factory() as session:
            row = self._get_row(session, transaction_id, user_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    # --- reads ---

    def get_transaction(self, transaction_id: str, user_id: str) -> dict[str, Any] | None:
        with self._session_factory() as session:
            row = self._get_row(session, transaction_id, user_id)
            return _to_dict(row) if row else None

    def list_transactions(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        start: date | None = None,
        end: date | None = None,
        category: str | None = None,
        type: str | None = None,
        query: str | None = None,
        min_amount: float | None = None,
        max_amount: float | None = None,
        order_by: str = "date",
    ) -> list[dict[str, Any]]:
        stmt = self._filtered(
            select(Transaction), user_id, start, end, category, type, query, min_amount, max_amount
        )
        if order_by == "amount":
            stmt = stmt.order_by(Transaction.amount.desc(), Transaction.date.desc())
        else:
            stmt = stmt.order_by(Transaction.date.desc(), Transaction.created_at.desc())
        stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)

        with self._session_factory() as session:
            return [_to_dict(row) for row in session.scalars(stmt).all()]

    def get_summary(
        self,
        user_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        is_expense = Transaction.type == "expense"
        totals_stmt = self._filtered(
            select(
                func.coalesce(func.sum(case((is_expense, Transaction.amount), else_=0)), 0),
                func.coalesce(func.sum(case((~is_expense, Transaction.amount), else_=0)), 0),
                func.count(Transaction.id),
            ),
            user_id, start, end, category,
        )
        category_stmt = self._filtered(
            select(Transaction.category, func.sum(Transaction.amount), func.count(Transaction.id)),
            user_id, start, end, category, type="expense",
        ).group_by(Transaction.category)

        with self._session_factory() as session:
            total_expense, total_income, count = session.execute(totals_stmt).one()
            rows = session.execute(category_stmt).all()

        by_category = {cat: _money(amount) for cat, amount, _ in rows}
        by_category = dict(sorted(by_category.items(), key=lambda kv: kv[1], reverse=True))
        expense, income = _money(total_expense), _money(total_income)
        return {
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None,
            "category": category,
            "total_expense": expense,
            "total_income": income,
            "net": _money(to_decimal(income) - to_decimal(expense)),
            "transaction_count": int(count),
            "by_category": by_category,
        }

    # --- helpers ---

    @staticmethod
    def _get_row(session, transaction_id: str, user_id: str) -> Transaction | None:
        stmt = select(Transaction).where(
            Transaction.id == transaction_id, Transaction.user_id == user_id
        )
        return session.scalar(stmt)

    @staticmethod
    def _filtered(
        stmt: Select,
        user_id: str,
        start: date | None = None,
        end: date | None = None,
        category: str | None = None,
        type: str | None = None,
        query: str | None = None,
        min_amount: float | None = None,
        max_amount: float | None = None,
    ) -> Select:
        stmt = stmt.where(Transaction.user_id == user_id)
        if start is not None:
            stmt = stmt.where(Transaction.date >= start)
        if end is not None:
            stmt = stmt.where(Transaction.date <= end)
        if category:
            stmt = stmt.where(Transaction.category == category)
        if type:
            stmt = stmt.where(Transaction.type == type)
        if query:
            stmt = stmt.where(func.lower(Transaction.description).contains(query.lower()))
        if min_amount is not None:
            stmt = stmt.where(Transaction.amount >= to_decimal(min_amount))
        if max_amount is not None:
            stmt = stmt.where(Transaction.amount <= to_decimal(max_amount))
        return stmt
