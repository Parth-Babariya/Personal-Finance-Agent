"""Database setup utilities for SQLAlchemy-backed services."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

Base = declarative_base()


def make_engine(database_url: str) -> Engine:
    """Create an engine; in-memory SQLite shares one connection across threads."""
    if database_url.startswith("sqlite"):
        kwargs: dict = {"connect_args": {"check_same_thread": False}}
        if database_url in {"sqlite://", "sqlite:///:memory:", "sqlite+pysqlite:///:memory:"}:
            kwargs["poolclass"] = StaticPool
        return create_engine(database_url, **kwargs)
    return create_engine(database_url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create tables if they do not exist (dev/test convenience; use Alembic in prod)."""
    from src.db import models  # noqa: F401  # Ensure model metadata is registered.

    Base.metadata.create_all(bind=engine)
