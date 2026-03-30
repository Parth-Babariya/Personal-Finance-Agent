"""SQLAlchemy-backed transaction service with user-level isolation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from ..interfaces.transaction_service import ITransactionService
from .models import Transaction


class SQLTransactionService(ITransactionService):
    """Persistent transaction service using a relational database."""

    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    def add_transaction(
        self, transaction: dict[str, Any], user_id: str | None = None
    ) -> dict[str, Any]:
        """Persist a user-scoped transaction and return normalized payload."""
        if user_id is None:
            raise ValueError("user_id is required for SQLTransactionService")

        with self._session_factory() as session:
            record = Transaction(
                user_id=user_id,
                amount=float(transaction["amount"]),
                category=transaction.get("category", "uncategorized"),
                description=transaction.get("description", ""),
                date=transaction.get("date", datetime.now(timezone.utc)),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return {
                "id": record.id,
                "user_id": record.user_id,
                "amount": record.amount,
                "category": record.category,
                "description": record.description,
                "date": record.date.isoformat(),
            }

    def get_transactions(
        self, limit: int = 50, offset: int = 0, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return paginated user transactions ordered by newest first."""
        if user_id is None:
            raise ValueError("user_id is required for SQLTransactionService")

        with self._session_factory() as session:
            stmt = (
                select(Transaction)
                .where(Transaction.user_id == user_id)
                .order_by(Transaction.date.desc())
                .offset(offset)
            )
            if limit != 0:
                stmt = stmt.limit(limit)

            rows = session.scalars(stmt).all()
            return [
                {
                    "id": row.id,
                    "user_id": row.user_id,
                    "amount": row.amount,
                    "category": row.category,
                    "description": row.description,
                    "date": row.date.isoformat(),
                }
                for row in rows
            ]

    def get_summary(self, period: str = "month", user_id: str | None = None) -> dict[str, Any]:
        """Compute aggregate total/count/by-category for one user."""
        if user_id is None:
            raise ValueError("user_id is required for SQLTransactionService")

        with self._session_factory() as session:
            total_stmt = select(
                func.coalesce(func.sum(Transaction.amount), 0.0),
                func.count(Transaction.id),
            ).where(Transaction.user_id == user_id)
            total_spending, transaction_count = session.execute(total_stmt).one()

            category_stmt = (
                select(Transaction.category, func.sum(Transaction.amount))
                .where(Transaction.user_id == user_id)
                .group_by(Transaction.category)
            )
            by_category = {
                category: float(amount or 0.0)
                for category, amount in session.execute(category_stmt).all()
            }

            return {
                "period": period,
                "total_spending": float(total_spending or 0.0),
                "transaction_count": int(transaction_count),
                "by_category": by_category,
            }
