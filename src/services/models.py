"""SQLAlchemy ORM models for users and transactions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class User(Base):
    """Authenticated user account."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    transactions: Mapped[list["Transaction"]] = relationship(back_populates="user")


class Transaction(Base):
    """A financial transaction scoped to a user."""

    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    amount: Mapped[float] = mapped_column(Float)
    category: Mapped[str] = mapped_column(String(64), default="uncategorized")
    description: Mapped[str] = mapped_column(Text, default="")
    date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    user: Mapped[User] = relationship(back_populates="transactions")
