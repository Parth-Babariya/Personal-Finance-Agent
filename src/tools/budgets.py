"""Budget tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from src.core.types import ExpenseCategory
from src.services.periods import resolve_period
from src.tools.base import Tool, ToolArgs, ToolContext


class BudgetArgs(ToolArgs):
    category: ExpenseCategory | None = Field(default=None, description="Omit to check every budget.")
    month: Literal["this_month", "last_month"] = "this_month"


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
