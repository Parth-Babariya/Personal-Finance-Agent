"""Abstract interface for generating financial insights.

Implementations can be rule-based, statistical, or LLM-backed; callers only
depend on this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IInsightGenerator(ABC):
    """Derives human-readable insights from a list of transactions."""

    @abstractmethod
    def generate_insights(self, transactions: list[dict[str, Any]]) -> list[str]:
        """Return insight strings for the given transactions."""
