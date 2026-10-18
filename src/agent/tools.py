"""Agent tools: typed, validated functions the LLM (or fallback planner) can call.

Each tool declares a Pydantic ``Args`` model. The model's JSON schema is sent to
the LLM, and the model's arguments are validated against it before execution.
``user_id`` is never an LLM-visible argument: it comes from the authenticated
request via ``ToolContext``, so the model cannot read another user's data.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from datetime import date
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..interfaces.insight_generator import IInsightGenerator
from ..interfaces.transaction_service import ITransactionService
from ..services.analytics import AnalyticsService
from ..services.budgets import BudgetService
from ..services.goals import GoalService
from ..services.models import CATEGORIES, EXPENSE_CATEGORIES
from ..services.pending_actions import PendingActionService
from ..services.periods import Period, describe_range, resolve_period

logger = logging.getLogger(__name__)

Category = Literal[CATEGORIES]  # type: ignore[valid-type]
ExpenseCategory = Literal[EXPENSE_CATEGORIES]  # type: ignore[valid-type]
TxType = Literal["expense", "income"]


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    today: date


@dataclass
class ToolDeps:
    """Services tools can use. Optional ones enable extra tools when provided."""

    tx: ITransactionService
    insights: IInsightGenerator
    actions: PendingActionService | None = None
    budgets: BudgetService | None = None
    analytics: AnalyticsService | None = None
    goals: GoalService | None = None


@dataclass
class ToolOutcome:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    ok: bool = True

    def to_record(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "ok": self.ok, "result": self.result}


# --- argument models ---


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RangeArgs(_Args):
    period: Period | None = Field(
        default="this_month",
        description="Named period. Ignored if start_date or end_date is given.",
    )
    start_date: date | None = Field(default=None, description="Inclusive start, YYYY-MM-DD.")
    end_date: date | None = Field(default=None, description="Inclusive end, YYYY-MM-DD.")

    @model_validator(mode="after")
    def _check_order(self) -> RangeArgs:
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        return self

    def resolve(self, today: date) -> tuple[date | None, date | None]:
        if self.start_date or self.end_date:
            return self.start_date, self.end_date
        return resolve_period(self.period or "all_time", today)


class SummaryArgs(RangeArgs):
    category: Category | None = Field(default=None, description="Restrict to one category.")


class CompareArgs(_Args):
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


class BudgetArgs(_Args):
    category: ExpenseCategory | None = Field(default=None, description="Omit to check every budget.")
    month: Literal["this_month", "last_month"] = "this_month"


class TrendArgs(_Args):
    months: int = Field(default=6, ge=2, le=12, description="How many months back, including this one.")
    category: Category | None = None


class UnusualArgs(RangeArgs):
    period: Period | None = Field(default="last_30_days", description="Window to scan for unusual expenses.")


class NoArgs(_Args):
    pass


class AddTransactionArgs(_Args):
    amount: float = Field(gt=0, le=10_000_000)
    type: TxType = "expense"
    category: Category = "other"
    description: str = Field(default="", max_length=200)
    date: dt.date | None = Field(default=None, description="Defaults to today. YYYY-MM-DD.")


# --- tools ---


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]
    mutates: ClassVar[bool] = False
    requires: ClassVar[tuple[str, ...]] = ()  # optional ToolDeps fields this tool needs

    def __init__(self, deps: ToolDeps):
        self._deps = deps
        self._tx = deps.tx
        self._insights = deps.insights
        self._actions = deps.actions

    @abstractmethod
    def run(self, args: Any, ctx: ToolContext) -> dict[str, Any]:
        """Execute synchronously (the registry runs it in a worker thread)."""

    def schema(self) -> dict[str, Any]:
        params = self.Args.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


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
            "transactions": [_compact(t) for t in matches[: args.limit]],
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
        return {"period": describe_range(start, end), "transactions": [_compact(t) for t in rows]}


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
        "Propose recording a new expense or income. Only call when the user clearly states an amount "
        "they spent or received. The transaction is NOT saved until the user confirms it."
    )
    Args = AddTransactionArgs
    mutates = True

    def run(self, args: AddTransactionArgs, ctx: ToolContext) -> dict[str, Any]:
        when = args.date or ctx.today
        if when > ctx.today:
            raise ValueError("Transaction date cannot be in the future")
        payload = {**args.model_dump(), "date": when.isoformat()}
        duplicates = [
            _compact(t)
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


class CheckBudget(Tool):
    name = "check_budget"
    description = (
        "Monthly budget usage per category: limit, spent, remaining, percent used, state "
        "(ok/warning/over) and projected month-end spend. Use for 'am I within budget' questions."
    )
    Args = BudgetArgs
    requires = ("budgets",)

    def run(self, args: BudgetArgs, ctx: ToolContext) -> dict[str, Any]:
        start, _ = resolve_period(args.month, ctx.today)
        budgets = self._deps.budgets.status(ctx.user_id, args.category, month=start)
        if not budgets:
            scope = f"for {args.category}" if args.category else "at all"
            return {"budgets": [], "note": f"The user has not set a budget {scope}."}
        return {"budgets": budgets}


def _compact(t: dict[str, Any]) -> dict[str, Any]:
    return {k: t[k] for k in ("id", "date", "type", "amount", "category", "description")}


class GetMonthlyTrend(Tool):
    name = "get_monthly_trend"
    description = (
        "Month-by-month expense, income and net totals (optionally for one category), plus the average "
        "monthly expense over complete months. Use for trends and 'am I spending more lately' questions."
    )
    Args = TrendArgs
    requires = ("analytics",)

    def run(self, args: TrendArgs, ctx: ToolContext) -> dict[str, Any]:
        return self._deps.analytics.monthly_trend(ctx.user_id, args.months, args.category)


class FindUnusualSpending(Tool):
    name = "find_unusual_spending"
    description = (
        "Expenses that are unusually large compared with the user's history in the same category or "
        "overall, each with a reason. Recurring bills are excluded."
    )
    Args = UnusualArgs
    requires = ("analytics",)

    def run(self, args: UnusualArgs, ctx: ToolContext) -> dict[str, Any]:
        start, end = args.resolve(ctx.today)
        unusual = self._deps.analytics.unusual_expenses(ctx.user_id, start, end)
        return {"period": describe_range(start, end), "unusual": [
            {**_compact(t), "reason": t["reason"]} for t in unusual
        ]}


class FindRecurringPayments(Tool):
    name = "find_recurring_payments"
    description = (
        "Detected subscriptions and recurring bills (same payee in at least 3 months with a stable amount), "
        "with typical amount, next expected date and estimated monthly total."
    )
    Args = NoArgs
    requires = ("analytics",)

    def run(self, args: NoArgs, ctx: ToolContext) -> dict[str, Any]:
        return self._deps.analytics.recurring_payments(ctx.user_id)


class GetSavingsGoals(Tool):
    name = "get_savings_goals"
    description = (
        "The user's savings goals: target, saved so far, progress, deadline, required saving per month, "
        "average monthly saving over recent months, whether they are on track and projected completion."
    )
    Args = NoArgs
    requires = ("goals",)

    def run(self, args: NoArgs, ctx: ToolContext) -> dict[str, Any]:
        goals = self._deps.goals.list(ctx.user_id)
        return {"goals": goals} if goals else {"goals": [], "note": "The user has no savings goals yet."}


DEFAULT_TOOLS: tuple[type[Tool], ...] = (
    GetSpendingSummary,
    ComparePeriods,
    SearchTransactions,
    GetTopExpenses,
    GetInsights,
    AddTransaction,
    CheckBudget,
    GetMonthlyTrend,
    FindUnusualSpending,
    FindRecurringPayments,
    GetSavingsGoals,
)


class ToolRegistry:
    """Holds tools, exposes their schemas, and executes calls safely."""

    def __init__(self, tools: list[Tool]):
        self._tools = {tool.name: tool for tool in tools}

    @classmethod
    def default(
        cls,
        tx: ITransactionService,
        insights: IInsightGenerator,
        actions: PendingActionService | None = None,
        **services: Any,
    ) -> ToolRegistry:
        """Build every tool whose required services are available."""
        known = {f.name for f in fields(ToolDeps)}
        unknown = set(services) - known
        if unknown:
            raise TypeError(f"Unknown services: {sorted(unknown)}")
        deps = ToolDeps(tx, insights, actions, **services)
        return cls([
            tool_cls(deps) for tool_cls in DEFAULT_TOOLS
            if all(getattr(deps, name) is not None for name in tool_cls.requires)
        ])

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def execute(self, name: str, raw_arguments: str | dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        """Validate and run a tool call. Never raises: errors are returned to the model."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutcome(name, {}, {"error": f"Unknown tool '{name}'"}, ok=False)

        try:
            data = json.loads(raw_arguments or "{}") if isinstance(raw_arguments, str) else raw_arguments
            if not isinstance(data, dict):
                raise ValueError("arguments must be a JSON object")
            args = tool.Args.model_validate(data)
        except (ValueError, ValidationError) as exc:
            return ToolOutcome(name, {}, {"error": f"Invalid arguments: {exc}"}, ok=False)

        arguments = args.model_dump(mode="json", exclude_none=True)
        try:
            result = await asyncio.to_thread(tool.run, args, ctx)
        except ValueError as exc:
            return ToolOutcome(name, arguments, {"error": str(exc)}, ok=False)
        except Exception:
            logger.exception("Tool %s failed", name)
            return ToolOutcome(name, arguments, {"error": "Internal error while running tool"}, ok=False)
        return ToolOutcome(name, arguments, result)
