"""SQL-backed conversation history (replaces the old in-process dict memory)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import sessionmaker

from .models import ChatMessage


def _to_dict(row: ChatMessage) -> dict[str, Any]:
    return {
        "id": row.id,
        "role": row.role,
        "content": row.content,
        "meta": row.meta or {},
        "created_at": row.created_at.isoformat(),
    }


class ConversationStore:
    """Persists chat turns per user so the agent can handle follow-up questions."""

    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    def add_message(
        self, user_id: str, role: str, content: str, meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        with self._session_factory() as session:
            row = ChatMessage(user_id=user_id, role=role, content=content, meta=meta)
            session.add(row)
            session.commit()
            return _to_dict(row)

    def recent(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return the last ``limit`` messages in chronological order."""
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.user_id == user_id)
            .order_by(ChatMessage.id.desc())
            .limit(limit)
        )
        with self._session_factory() as session:
            rows = session.scalars(stmt).all()
        return [_to_dict(row) for row in reversed(rows)]

    def search(self, user_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        stmt = (
            select(ChatMessage)
            .where(
                ChatMessage.user_id == user_id,
                func.lower(ChatMessage.content).contains(query.lower()),
            )
            .order_by(ChatMessage.id.desc())
            .limit(limit)
        )
        with self._session_factory() as session:
            return [_to_dict(row) for row in session.scalars(stmt).all()]

    def clear(self, user_id: str) -> int:
        with self._session_factory() as session:
            result = session.execute(delete(ChatMessage).where(ChatMessage.user_id == user_id))
            session.commit()
            return result.rowcount or 0
