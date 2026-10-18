"""Rule-based financial insight generator."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.interfaces.insight_generator import IInsightGenerator

LARGE_TRANSACTION_SHARE = 0.25  # a single expense above 25% of spend is notable
HIGH_CATEGORY_SHARE = 0.40  # one category above 40% of spend is notable


class SimpleInsightGenerator(IInsightGenerator):
    """Deterministic heuristics over a set of transactions (no LLM required)."""

    def __init__(self, currency_symbol: str = "₹"):
        self._cur = currency_symbol

    def _fmt(self, amount: float) -> str:
        return f"{self._cur}{amount:,.2f}"

    def generate_insights(self, transactions: list[dict[str, Any]]) -> list[str]:
        if not transactions:
            return ["No transactions to analyze."]

        expenses = [t for t in transactions if t.get("type", "expense") == "expense"]
        income = sum(t["amount"] for t in transactions if t.get("type") == "income")
        spent = sum(t["amount"] for t in expenses)
        insights: list[str] = []

        if expenses:
            by_category: dict[str, float] = defaultdict(float)
            for t in expenses:
                by_category[t.get("category", "other")] += t["amount"]
            top_cat, top_amt = max(by_category.items(), key=lambda kv: kv[1])
            share = top_amt / spent if spent else 0
            insights.append(
                f"Your highest spending category is '{top_cat}' at {self._fmt(top_amt)} "
                f"({share:.0%} of expenses)."
            )
            if share > HIGH_CATEGORY_SHARE and len(by_category) > 1:
                insights.append(f"'{top_cat}' dominates your spending; consider setting a budget for it.")

            largest = max(expenses, key=lambda t: t["amount"])
            if spent and largest["amount"] / spent > LARGE_TRANSACTION_SHARE and len(expenses) > 1:
                label = largest.get("description") or largest.get("category", "")
                insights.append(
                    f"Your largest expense was {self._fmt(largest['amount'])} ({label}) on {largest['date']}."
                )

        if income:
            savings_rate = (income - spent) / income
            if savings_rate < 0:
                insights.append(f"You spent {self._fmt(spent - income)} more than you earned.")
            else:
                insights.append(f"You saved {savings_rate:.0%} of your income ({self._fmt(income - spent)}).")

        insights.append(f"Total spending: {self._fmt(spent)} across {len(expenses)} expenses.")
        return insights
