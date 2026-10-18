"""SQLAlchemy ORM models and shared domain constants."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, Date, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

TRANSACTION_TYPES = ("expense", "income")

EXPENSE_CATEGORIES = (
    "food",
    "groceries",
    "transport",
    "shopping",
    "entertainment",
    "utilities",
    "rent",
    "health",
    "education",
    "travel",
    "subscriptions",
    "other",
)
INCOME_CATEGORIES = ("salary", "freelance", "investment", "gift", "refund")
CATEGORIES = EXPENSE_CATEGORIES + INCOME_CATEGORIES


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    """Authenticated user account."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    transactions: Mapped[list[Transaction]] = relationship(back_populates="user")


class Transaction(Base):
    """A financial transaction (expense or income) scoped to a user."""

    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    type: Mapped[str] = mapped_column(String(16), default="expense", index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    category: Mapped[str] = mapped_column(String(32), default="other", index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    date: Mapped[date] = mapped_column(Date, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped[User] = relationship(back_populates="transactions")


class ChatMessage(Base):
    """One persisted turn of a user's conversation with the agent."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
