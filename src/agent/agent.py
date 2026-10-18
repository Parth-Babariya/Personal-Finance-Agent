"""Tool-calling finance agent with streaming, history and a deterministic fallback.

Loop: send system prompt + recent history + user message to the LLM with the
tool schemas. If the model asks for tools, run them (validated, user-scoped),
append the results and call the model again. Stop when it answers in text or
the step budget is exhausted. If no LLM is configured, or it fails before
producing output, the keyword fallback planner answers instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..services.conversation_store import ConversationStore
from .fallback import KeywordFallbackPlanner
from .llm import LLMClient, LLMResponse, TextDelta, Usage
from .prompts import TOOL_BUDGET_EXHAUSTED, build_system_prompt
from .tools import ToolContext, ToolOutcome, ToolRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS_PER_STEP = 6


@dataclass
class AgentResult:
    reply: str
    mode: str  # "llm" | "fallback"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0


@dataclass
class _RunState:
    outcomes: list[ToolOutcome] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    steps: int = 0
    reply: str = ""
    emitted_text: bool = False


class FinanceAgent:
    def __init__(
        self,
        registry: ToolRegistry,
        conversations: ConversationStore,
        llm: LLMClient | None,
        *,
        currency: str = "₹",
        max_steps: int = 5,
        history_limit: int = 10,
        max_message_chars: int = 2000,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._registry = registry
        self._conversations = conversations
        self._llm = llm
        self._currency = currency
        self._max_steps = max_steps
        self._history_limit = history_limit
        self._max_message_chars = max_message_chars
        self._today = today
        self._fallback = KeywordFallbackPlanner(registry, currency)

    @property
    def llm_enabled(self) -> bool:
        return self._llm is not None

    async def respond(self, message: str, user_id: str) -> AgentResult:
        """Run the agent to completion and return the final result."""
        done: dict[str, Any] = {}
        async for event in self.stream(message, user_id):
            if event["type"] == "done":
                done = event
        return AgentResult(
            reply=done["reply"],
            mode=done["mode"],
            tool_calls=done["tool_calls"],
            steps=done["steps"],
            usage=Usage(**done["usage"]),
            latency_ms=done["latency_ms"],
        )

    async def stream(self, message: str, user_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield events: ``token``, ``tool_call``, ``tool_result`` and finally ``done``."""
        message = message.strip()
        if not message:
            raise ValueError("Message must not be empty")
        message = message[: self._max_message_chars]

        started = time.perf_counter()
        ctx = ToolContext(user_id=user_id, today=self._today())
        history = await asyncio.to_thread(self._conversations.recent, user_id, self._history_limit)
        state = _RunState()
        mode = "fallback"

        if self._llm is not None:
            mode = "llm"
            try:
                async for event in self._run_llm(message, history, ctx, state):
                    yield event
            except Exception:
                logger.exception("LLM run failed")
                wrote_data = any(
                    o.ok and getattr(self._registry.get(o.name), "mutates", False) for o in state.outcomes
                )
                if state.emitted_text or wrote_data:
                    state.reply = (state.reply + "\n\n(The response was interrupted. Please try again.)").strip()
                else:
                    mode, state = "fallback", _RunState()

        if mode == "fallback":
            reply, outcomes = await self._fallback.respond(message, ctx, history)
            state.outcomes, state.reply = outcomes, reply
            for outcome in outcomes:
                yield {"type": "tool_call", "name": outcome.name, "arguments": outcome.arguments}
                yield {"type": "tool_result", "name": outcome.name, "ok": outcome.ok}
            yield {"type": "token", "text": reply}

        records = [o.to_record() for o in state.outcomes]
        await asyncio.to_thread(self._conversations.add_message, user_id, "user", message)
        await asyncio.to_thread(
            self._conversations.add_message,
            user_id,
            "assistant",
            state.reply,
            {"mode": mode, "tools": [r["name"] for r in records]},
        )

        yield {
            "type": "done",
            "reply": state.reply,
            "mode": mode,
            "tool_calls": records,
            "steps": state.steps,
            "usage": {"prompt_tokens": state.usage.prompt_tokens, "completion_tokens": state.usage.completion_tokens},
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    async def _run_llm(
        self, message: str, history: list[dict[str, Any]], ctx: ToolContext, state: _RunState
    ) -> AsyncIterator[dict[str, Any]]:
        assert self._llm is not None
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(ctx.today, self._currency)},
            *({"role": m["role"], "content": m["content"]} for m in history),
            {"role": "user", "content": message},
        ]
        tools = self._registry.schemas()

        for step in range(self._max_steps + 1):
            budget_left = step < self._max_steps
            if not budget_left:
                messages.append({"role": "system", "content": TOOL_BUDGET_EXHAUSTED})

            response: LLMResponse | None = None
            async for chunk in self._llm.stream(messages, tools if budget_left else []):
                if isinstance(chunk, TextDelta):
                    state.emitted_text = True
                    yield {"type": "token", "text": chunk.text}
                else:
                    response = chunk
            if response is None:
                raise RuntimeError("LLM stream ended without a final response")

            state.steps += 1
            state.usage.add(response.usage)

            if not response.tool_calls or not budget_left:
                state.reply = response.content.strip()
                return

            messages.append({
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                    for c in response.tool_calls
                ],
            })
            for index, call in enumerate(response.tool_calls):
                if index >= MAX_TOOL_CALLS_PER_STEP:
                    # Every tool_call id needs a reply, so refuse the excess explicitly.
                    outcome = ToolOutcome(call.name, {}, {"error": "Too many tool calls in one step"}, ok=False)
                else:
                    yield {"type": "tool_call", "name": call.name, "arguments": _safe_json(call.arguments)}
                    outcome = await self._registry.execute(call.name, call.arguments, ctx)
                    state.outcomes.append(outcome)
                    yield {"type": "tool_result", "name": call.name, "ok": outcome.ok}
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(outcome.result, default=str),
                })


def _safe_json(raw: str) -> Any:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
