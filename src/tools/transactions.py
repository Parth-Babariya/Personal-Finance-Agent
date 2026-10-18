"""Tools over the user's transactions: summaries, comparisons, search, insights and writes."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import Field

from src.core.types import Category, TxType
from src.services.periods import Period, describe_range, resolve_period
from src.tools.base import RangeArgs, Tool, ToolArgs, ToolContext, compact


class SummaryArgs(RangeArgs):
    category: Category | None = Field(default=None, description="Restrict to one category.")


class CompareArgs(ToolArgs):
    period_a: Period = Field(description="The period of interest, e.g. this_month.")
    period_b: Period = Field(description="The baseline period, e.g. last_month.")
    category: Category | None = None


class SearchArgs(RangeArgs):
    period: Period | None = Field(default="all_time", description="Named period.")
    query: str | None = Field(default=None, max_length=100, description="Text to match in descriptions.")
    category: Category | None = None
    type: TxType | None = None
    min_amount: float | None = Field(default=None, ge=0)
    max_amount: float | None = Field(default=None, ge=0)
    limit: int = Field(default=20, ge=1, le=50)


class TopExpensesArgs(RangeArgs):
    category: Category | None = None
    limit: int = Field(default=5, ge=1, le=20)


class AddTransactionArgs(ToolArgs):
    amount: float = Field(gt=0, le=10_000_000)
    type: TxType = "expense"
    category: Category = "other"
    description: str = Field(default="", max_length=200)
    date: dt.date | None = Field(default=None, description="Defaults to today. YYYY-MM-DD.")


class GetSpendingSummary(Tool):
    name = "get_spending_summary"
    description = (
        "Total expenses, income, net and per-category expense totals for a period, "
        "optionally for one category. Use for 'how much did I spend/earn' questions."
    )
    Args = SummaryArgs

    def run(self, args: SummaryArgs, ctx: ToolContext) -> dict[str, Any]:
        start, end = args.resolve(ctx.today)
        summary = self._tx.get_summary(ctx.user_id, start=start, end=end, category=args.category)
        return {"period": describe_range(start, end), **summary}


class ComparePeriods(Tool):
    name = "compare_periods"
    description = "Compare spending between two named periods, overall and per category."
    Args = CompareArgs

    def run(self, args: CompareArgs, ctx: ToolContext) -> dict[str, Any]:
        summaries = {}
        for key, period in (("a", args.period_a), ("b", args.period_b)):
            start, end = resolve_period(period, ctx.today)
            summaries[key] = {
                "period": period,
                "range": describe_range(start, end),
                **self._tx.get_summary(ctx.user_id, start=start, end=end, category=args.category),
            }
        a, b = summaries["a"], summaries["b"]
        change = round(a["total_expense"] - b["total_expense"], 2)
        pct = round(change / b["total_expense"] * 100, 1) if b["total_expense"] else None
        categories = sorted(set(a["by_category"]) | set(b["by_category"]))
        per_category = {
            cat: {
                "a": a["by_category"].get(cat, 0.0),
                "b": b["by_category"].get(cat, 0.0),
                "change": round(a["by_category"].get(cat, 0.0) - b["by_category"].get(cat, 0.0), 2),
            }
            for cat in categories
        }
        return {"a": a, "b": b, "expense_change": change, "expense_change_pct": pct, "by_category": per_category}


class SearchTransactions(Tool):
    name = "search_transactions"
    description = "Find individual transactions by text, category, type, amount range and dates."
    Args = SearchArgs

    def run(self, args: SearchArgs, ctx: ToolContext) -> dict[str, Any]:
        start, end = args.resolve(ctx.today)
        matches = self._tx.list_transactions(
            ctx.user_id,
            limit=0,
            start=start,
            end=end,
            category=args.category,
            type=args.type,
            query=args.query,
            min_amount=args.min_amount,
            max_amount=args.max_amount,
        )
        return {
            "period": describe_range(start, end),
            "match_count": len(matches),
            "total_amount": round(sum(t["amount"] for t in matches), 2),
            "transactions": [compact(t) for t in matches[: args.limit]],
            "truncated": len(matches) > args.limit,
        }


class GetTopExpenses(Tool):
    name = "get_top_expenses"
    description = "The largest individual expenses in a period, biggest first."
    Args = TopExpensesArgs

    def run(self, args: TopExpensesArgs, ctx: ToolContext) -> dict[str, Any]:
        start, end = args.resolve(ctx.today)
        rows = self._tx.list_transactions(
            ctx.user_id, limit=args.limit, start=start, end=end,
            category=args.category, type="expense", order_by="amount",
        )
        return {"period": describe_range(start, end), "transactions": [compact(t) for t in rows]}


class GetInsights(Tool):
    name = "get_insights"
    description = "Rule-based observations about spending habits and savings for a period."
    Args = RangeArgs

    def run(self, args: RangeArgs, ctx: ToolContext) -> dict[str, Any]:
        start, end = args.resolve(ctx.today)
        rows = self._tx.list_transactions(ctx.user_id, limit=0, start=start, end=end)
        return {"period": describe_range(start, end), "insights": self._insights.generate_insights(rows)}


class AddTransaction(Tool):
    name = "add_transaction"
    description = (
        "Propose recording a new expense or income. Call it directly when the user states an amount they "
        "spent or received; it does not save anything. The app asks the user to confirm, and the result "
        "lists possible duplicates, so do not ask first or search for duplicates yourself."
    )
    Args = AddTransactionArgs
    mutates = True

    def run(self, args: AddTransactionArgs, ctx: ToolContext) -> dict[str, Any]:
        when = args.date or ctx.today
        if when > ctx.today:
            raise ValueError("Transaction date cannot be in the future")
        payload = {**args.model_dump(), "date": when.isoformat()}
        duplicates = [
            compact(t)
            for t in self._tx.list_transactions(
                ctx.user_id, start=when, end=when, category=args.category,
                min_amount=args.amount, max_amount=args.amount,
            )
        ]

        budgets = self._deps.budgets
        impact = None
        if budgets is not None and args.type == "expense":
            impact = budgets.impact(ctx.user_id, args.category, args.amount, when)

        if self._actions is None:  # confirmation disabled: write immediately
            added = self._tx.add_transaction({**payload, "date": when}, ctx.user_id)
            return {"status": "saved", "added": added, "possible_duplicates": duplicates, "budget_impact": impact}

        action = self._actions.propose(ctx.user_id, "add_transaction", payload)
        return {
            "status": "pending_confirmation",
            "action_id": action["id"],
            "transaction": payload,
            "possible_duplicates": duplicates,
            "budget_impact": impact,
            "instructions": (
                "Not saved yet. Tell the user exactly what will be recorded and ask them to confirm "
                "(reply 'yes' or press Confirm). Mention possible duplicates, and warn if budget_impact "
                "shows the budget would reach its warning level or be exceeded."
            ),
        }
