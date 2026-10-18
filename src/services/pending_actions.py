"""Pending actions: writes proposed by the agent that need explicit user confirmation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from ..interfaces.transaction_service import ITransactionService
from .budgets import BudgetService
from .models import PendingAction


class ActionError(Exception):
    """Raised when an action cannot be confirmed or cancelled (expired or already resolved)."""


def _aware(value: datetime) -> datetime:
    # SQLite returns naive datetimes even for timezone-aware columns.
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _to_dict(row: PendingAction) -> dict[str, Any]:
    return {
        "id": row.id,
        "kind": row.kind,
        "payload": row.payload,
        "status": row.status,
        "created_at": _aware(row.created_at).isoformat(),
        "expires_at": _aware(row.expires_at).isoformat(),
        "result": row.result,
    }


class PendingActionService:
    def __init__(
        self,
        session_factory: sessionmaker,
        tx_service: ITransactionService,
        ttl_minutes: int = 15,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        budgets: BudgetService | None = None,
    ):
        self._session_factory = session_factory
        self._tx = tx_service
        self._budgets = budgets
        self._ttl = timedelta(minutes=ttl_minutes)
        self._now = now

    def propose(self, user_id: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = self._now()
        with self._session_factory() as session:
            row = PendingAction(
                user_id=user_id, kind=kind, payload=payload, status="pending",
                created_at=now, expires_at=now + self._ttl,
            )
            session.add(row)
            session.commit()
            return _to_dict(row)

    def list_open(self, user_id: str) -> list[dict[str, Any]]:
        now = self._now()
        stmt = (
            select(PendingAction)
            .where(PendingAction.user_id == user_id, PendingAction.status == "pending")
            .order_by(PendingAction.created_at)
        )
        with self._session_factory() as session:
            rows = session.scalars(stmt).all()
            for row in rows:
                if _aware(row.expires_at) <= now:
                    row.status = "expired"
            session.commit()
            return [_to_dict(row) for row in rows if row.status == "pending"]

    def confirm(self, action_id: str, user_id: str) -> dict[str, Any] | None:
        with self._session_factory() as session:
            row = self._open_row(session, action_id, user_id)
            if row is None:
                return None
            row.result = self._execute(row.kind, row.payload, user_id)
            row.status = "confirmed"
            row.resolved_at = self._now()
            session.commit()
            return _to_dict(row)

    def cancel(self, action_id: str, user_id: str) -> dict[str, Any] | None:
        with self._session_factory() as session:
            row = self._open_row(session, action_id, user_id)
            if row is None:
                return None
            row.status = "cancelled"
            row.resolved_at = self._now()
            session.commit()
            return _to_dict(row)

    def _open_row(self, session, action_id: str, user_id: str) -> PendingAction | None:
        row = session.scalar(
            select(PendingAction).where(PendingAction.id == action_id, PendingAction.user_id == user_id)
        )
        if row is None:
            return None
        if row.status == "pending" and _aware(row.expires_at) <= self._now():
            row.status = "expired"
            session.commit()
        if row.status != "pending":
            raise ActionError(f"Action is already {row.status}")
        return row

    def _execute(self, kind: str, payload: dict[str, Any], user_id: str) -> dict[str, Any]:
        if kind == "add_transaction":
            data = {**payload, "date": date.fromisoformat(payload["date"])}
            added = self._tx.add_transaction(data, user_id)
            alert = None
            if self._budgets is not None and added["type"] == "expense":
                alert = self._budgets.alert_for(user_id, added["category"], data["date"])
            return {"added": added, "budget_alert": alert}
        raise ActionError(f"Unknown action kind '{kind}'")
