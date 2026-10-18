"""Savings goal tools (read-only)."""

from __future__ import annotations

from typing import Any

from src.tools.base import NoArgs, Tool, ToolContext


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
