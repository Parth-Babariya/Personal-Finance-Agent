"""Abstract interface for transaction data access.

API routes and agent tools depend on this contract rather than on a concrete
store, so the persistence layer can change without touching business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any


class ITransactionService(ABC):
    """Contract for services that manage a user's financial transactions."""

    @abstractmethod
    def add_transaction(self, data: dict[str, Any], user_id: str) -> dict[str, Any]:
        """Persist a transaction and return it with server-side fields filled in."""

    @abstractmethod
    def get_transaction(self, transaction_id: str, user_id: str) -> dict[str, Any] | None:
        """Return one transaction, or ``None`` if it does not exist for this user."""

    @abstractmethod
    def update_transaction(
        self, transaction_id: str, changes: dict[str, Any], user_id: str
    ) -> dict[str, Any] | None:
        """Apply a partial update; ``None`` if the transaction was not found."""

    @abstractmethod
    def delete_transaction(self, transaction_id: str, user_id: str) -> bool:
        """Delete a transaction; ``False`` if it was not found."""

    @abstractmethod
    def list_transactions(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        start: date | None = None,
        end: date | None = None,
        category: str | None = None,
        type: str | None = None,
        query: str | None = None,
        min_amount: float | None = None,
        max_amount: float | None = None,
        order_by: str = "date",
    ) -> list[dict[str, Any]]:
        """Return filtered transactions, newest first (or largest first)."""

    @abstractmethod
    def get_summary(
        self,
        user_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate income, expenses, net and per-category expense totals."""
