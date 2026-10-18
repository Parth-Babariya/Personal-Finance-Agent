"""Analytics tools: monthly trends, unusual expenses and recurring payments."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from src.core.types import Category
from src.services.periods import Period, describe_range
from src.tools.base import NoArgs, RangeArgs, Tool, ToolArgs, ToolContext, compact


class TrendArgs(ToolArgs):
    months: int = Field(default=6, ge=2, le=12, description="How many months back, including this one.")
    category: Category | None = None


class UnusualArgs(RangeArgs):
    period: Period | None = Field(default="last_30_days", description="Window to scan for unusual expenses.")


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
            {**compact(t), "reason": t["reason"]} for t in unusual
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
