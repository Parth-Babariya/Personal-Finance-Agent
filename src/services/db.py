"""Database setup utilities for SQLAlchemy-backed services."""

from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./finance_agent.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def init_db() -> None:
    """Create database tables if they do not exist."""
    from . import models  # noqa: F401  # Ensure model metadata is registered.

    Base.metadata.create_all(bind=engine)
