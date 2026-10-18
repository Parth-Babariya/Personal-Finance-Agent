"""SQLAlchemy ORM models."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, Date, DateTime, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.session import Base


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


class PendingAction(Base):
    """A write proposed by the agent, applied only after the user confirms it."""

    __tablename__ = "pending_actions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DailyUsage(Base):
    """LLM tokens consumed per user per day."""

    __tablename__ = "daily_usage"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    requests: Mapped[int] = mapped_column(Integer, default=0)


class Budget(Base):
    """Monthly spending limit for one expense category."""

    __tablename__ = "budgets"
    __table_args__ = (UniqueConstraint("user_id", "category", name="uq_budget_user_category"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    category: Mapped[str] = mapped_column(String(32))
    monthly_limit: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SavingsGoal(Base):
    """A savings target, optionally with a deadline."""

    __tablename__ = "savings_goals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    target_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    saved_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=0)
    target_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ImportBatch(Base):
    """A parsed bank statement waiting for review before its rows are saved."""

    __tablename__ = "import_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    filename: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | committed | discarded
    rows: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    errors: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    imported_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
