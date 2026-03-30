"""Unit tests for SQLTransactionService guard clauses and user-scoped behavior."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.services.db import Base
from src.services.models import User
from src.services.sql_transaction_service import SQLTransactionService


class TestSQLTransactionService:
    """Covers error branches and persistence path in SQL transaction service."""

    def setup_method(self):
        engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            future=True,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(bind=engine)
        self.session_factory = sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
            future=True,
        )
        self.service = SQLTransactionService(self.session_factory)

        with self.session_factory() as session:
            user = User(username="sql_user", password_hash="dummy")
            session.add(user)
            session.commit()
            session.refresh(user)
            self.user_id = user.id

    def test_add_transaction_requires_user(self):
        """add_transaction should reject missing user_id."""
        with pytest.raises(ValueError):
            self.service.add_transaction({"amount": 10.0})

    def test_get_transactions_requires_user(self):
        """get_transactions should reject missing user_id."""
        with pytest.raises(ValueError):
            self.service.get_transactions()

    def test_get_summary_requires_user(self):
        """get_summary should reject missing user_id."""
        with pytest.raises(ValueError):
            self.service.get_summary()

    def test_user_scoped_read_paths(self):
        """Persisted records should be retrievable and summarized per user."""
        saved = self.service.add_transaction(
            {"amount": 55.0, "category": "food", "description": "meal"},
            user_id=self.user_id,
        )
        assert saved["user_id"] == self.user_id

        txns = self.service.get_transactions(user_id=self.user_id)
        assert len(txns) == 1
        assert txns[0]["amount"] == 55.0

        summary = self.service.get_summary(user_id=self.user_id)
        assert summary["total_spending"] == 55.0
        assert summary["transaction_count"] == 1
