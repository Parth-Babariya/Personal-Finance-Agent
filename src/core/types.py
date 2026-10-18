"""Domain vocabulary shared by the database, API, tools and workflows."""

from __future__ import annotations

from typing import Literal

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

Category = Literal[CATEGORIES]  # type: ignore[valid-type]
ExpenseCategory = Literal[EXPENSE_CATEGORIES]  # type: ignore[valid-type]
TxType = Literal["expense", "income"]
