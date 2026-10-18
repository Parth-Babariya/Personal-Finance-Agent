"""Agent-callable tools, grouped by domain.

- ``base``: ToolContext, ToolDeps, ToolOutcome, argument models and the Tool base class
- ``registry``: the ToolRegistry that validates and executes calls
- ``transactions``, ``budgets``, ``analytics``, ``goals``: the tools themselves
"""

from src.tools.base import Tool, ToolContext, ToolDeps, ToolOutcome
from src.tools.registry import DEFAULT_TOOLS, ToolRegistry

__all__ = ["DEFAULT_TOOLS", "Tool", "ToolContext", "ToolDeps", "ToolOutcome", "ToolRegistry"]
