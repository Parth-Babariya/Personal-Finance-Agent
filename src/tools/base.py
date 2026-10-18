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
        params = compact_schema(self.Args.model_json_schema())
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


def compact_schema(node: Any) -> Any:
    """Drop JSON-schema noise that costs tokens on every LLM call but tells the model nothing.

    - auto-generated ``title`` fields (they repeat the property name)
    - ``default: null`` (optional fields are already absent from ``required``)
    - ``anyOf: [X, {"type": "null"}]`` wrappers, collapsed to ``X``
    Descriptions, enums, ranges and defaults with real values are kept, and
    arguments are still validated against the full Pydantic model.
    """
    if isinstance(node, list):
        return [compact_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    variants = node.get("anyOf")
    if isinstance(variants, list) and len(variants) == 2 and {"type": "null"} in variants:
        inner = next(v for v in variants if v != {"type": "null"})
        node = {**{k: v for k, v in node.items() if k != "anyOf"}, **inner}

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title" and isinstance(value, str):
            continue
        if key == "default" and value is None:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: compact_schema(prop) for name, prop in value.items()}  # keys are field names
        else:
            out[key] = compact_schema(value)
    return out


def compact(t: dict[str, Any]) -> dict[str, Any]:
    """The transaction fields worth showing to the model."""
    return {k: t[k] for k in ("id", "date", "type", "amount", "category", "description")}
