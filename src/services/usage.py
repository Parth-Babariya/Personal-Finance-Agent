"""Per-user daily LLM token accounting, used to cap spend per user."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from src.db.models import DailyUsage


class UsageService:
    def __init__(self, session_factory: sessionmaker, today: Callable[[], date] = date.today):
        self._session_factory = session_factory
        self._today = today

    def record(self, user_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        day = self._today()
        with self._session_factory() as session:
            row = session.get(DailyUsage, (user_id, day))
            if row is None:
                row = DailyUsage(user_id=user_id, day=day, prompt_tokens=0, completion_tokens=0, requests=0)
                session.add(row)
            row.prompt_tokens += prompt_tokens
            row.completion_tokens += completion_tokens
            row.requests += 1
            session.commit()

    def tokens_today(self, user_id: str) -> int:
        stmt = select(DailyUsage).where(DailyUsage.user_id == user_id, DailyUsage.day == self._today())
        with self._session_factory() as session:
            row = session.scalar(stmt)
            return (row.prompt_tokens + row.completion_tokens) if row else 0
