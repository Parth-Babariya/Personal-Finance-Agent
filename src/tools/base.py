"""Building blocks for agent tools: context, dependencies, outcomes and the Tool base class.

Each tool declares a Pydantic ``Args`` model. Its JSON schema is sent to the
LLM, and the model's arguments are validated against it before execution.
``user_id`` is never an LLM-visible argument: it comes from the authenticated
request via ``ToolContext``, so the model cannot read another user's data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.services.periods import Period, resolve_period

if TYPE_CHECKING:
    from src.interfaces.insight_generator import IInsightGenerator
    from src.interfaces.transaction_service import ITransactionService
    from src.services.analytics import AnalyticsService
    from src.services.budgets import BudgetService
    from src.services.goals import GoalService
    from src.workflows.approvals import PendingActionService


@dataclass(frozen=True)
class ToolContext:
    """Per-request facts tools may rely on; never controlled by the model."""

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


# --- argument models shared by several tools ---


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(ToolArgs):
    pass


class RangeArgs(ToolArgs):
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


# --- base class ---


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


def compact(t: dict[str, Any]) -> dict[str, Any]:
    """The transaction fields worth showing to the model."""
    return {k: t[k] for k in ("id", "date", "type", "amount", "category", "description")}
