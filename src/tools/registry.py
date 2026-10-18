"""The tool registry: which tools exist, their schemas, and safe execution."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import fields
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from src.tools.analytics import FindRecurringPayments, FindUnusualSpending, GetMonthlyTrend
from src.tools.base import Tool, ToolContext, ToolDeps, ToolOutcome
from src.tools.budgets import CheckBudget
from src.tools.goals import GetSavingsGoals
from src.tools.transactions import (
    AddTransaction,
    ComparePeriods,
    GetInsights,
    GetSpendingSummary,
    GetTopExpenses,
    SearchTransactions,
)

if TYPE_CHECKING:
    from src.interfaces.insight_generator import IInsightGenerator
    from src.interfaces.transaction_service import ITransactionService
    from src.workflows.approvals import PendingActionService

logger = logging.getLogger(__name__)

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
