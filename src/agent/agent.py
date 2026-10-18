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
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..services.conversation_store import ConversationStore
from ..services.pending_actions import ActionError, PendingActionService
from .fallback import KeywordFallbackPlanner
from .llm import LLMClient, LLMResponse, TextDelta, Usage
from .prompts import TOOL_BUDGET_EXHAUSTED, build_system_prompt
from .redaction import redact
from .tools import ToolContext, ToolOutcome, ToolRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS_PER_STEP = 6
MAX_WRITES_PER_TURN = 3
INTERRUPTED_NOTE = "\n\n(The response was interrupted. Please try again.)"

# Short replies that confirm or cancel open pending actions without calling the LLM. A reply
# qualifies only if every word is in the vocabulary and at least one word is decisive.
_CONFIRM_WORDS = {"yes", "y", "yeah", "yep", "yup", "confirm", "confirmed", "ok", "okay", "sure", "save"}
_CANCEL_WORDS = {"no", "n", "nope", "cancel", "discard", "dont", "don't", "stop", "not"}
_FILLER_WORDS = {"it", "that", "this", "please", "go", "ahead", "do", "pls", "thanks"}


def _classify_reply(message: str) -> str | None:
    words = re.findall(r"[a-z']+", message.lower())
    if not words or len(words) > 6 or not set(words) <= _CONFIRM_WORDS | _CANCEL_WORDS | _FILLER_WORDS:
        return None
    if set(words) & _CANCEL_WORDS:
        return "cancel"
    if set(words) & _CONFIRM_WORDS or words in (["go", "ahead"], ["do", "it"]):
        return "confirm"
    return None


@dataclass
class AgentResult:
    reply: str
    mode: str  # "llm" | "fallback" | "action"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    redacted: list[str] = field(default_factory=list)
    fallback_reason: str | None = None  # "no_llm" | "llm_error" | "timeout"
    pending_actions: list[dict[str, Any]] = field(default_factory=list)


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
        actions: PendingActionService | None = None,
        currency: str = "₹",
        max_steps: int = 5,
        history_limit: int = 10,
        max_message_chars: int = 2000,
        timeout_seconds: float = 45.0,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._registry = registry
        self._conversations = conversations
        self._llm = llm
        self._actions = actions
        self._currency = currency
        self._max_steps = max_steps
        self._history_limit = history_limit
        self._max_message_chars = max_message_chars
        self._timeout = timeout_seconds
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
            redacted=done["redacted"],
            fallback_reason=done["fallback_reason"],
            pending_actions=done["pending_actions"],
        )

    async def stream(self, message: str, user_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield events: ``token``, ``tool_call``, ``tool_result`` and finally ``done``."""
        message = message.strip()
        if not message:
            raise ValueError("Message must not be empty")
        # Mask card numbers, Aadhaar, PAN, etc. before the LLM or the database sees them.
        masked = redact(message[: self._max_message_chars])
        message = masked.text

        started = time.perf_counter()
        ctx = ToolContext(user_id=user_id, today=self._today())
        history = await asyncio.to_thread(self._conversations.recent, user_id, self._history_limit)
        state = _RunState()
        mode, fallback_reason = "fallback", "no_llm"

        decision = await self._resolve_pending(message, user_id)
        if decision is not None:
            mode, fallback_reason = "action", None
            state.reply, state.outcomes = decision
            yield {"type": "token", "text": state.reply}
        elif self._llm is not None:
            mode, fallback_reason = "llm", None
            deadline = time.monotonic() + self._timeout
            try:
                async for event in _with_deadline(self._run_llm(message, history, ctx, state), deadline):
                    yield event
            except Exception as exc:
                fallback_reason = "timeout" if isinstance(exc, TimeoutError) else "llm_error"
                logger.warning("LLM run failed (%s)", fallback_reason, exc_info=fallback_reason == "llm_error")
                if state.emitted_text or self._wrote_data(state):
                    # Too late to switch: the user already saw text or data was written.
                    state.reply = (state.reply + INTERRUPTED_NOTE).strip()
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
            "redacted": masked.found,
            "fallback_reason": fallback_reason if mode == "fallback" else None,
            "pending_actions": [
                {
                    "id": o.result["action_id"],
                    "kind": o.name,
                    "payload": o.result["transaction"],
                    "possible_duplicates": o.result["possible_duplicates"],
                }
                for o in state.outcomes
                if o.ok and o.result.get("status") == "pending_confirmation"
            ],
        }

    def _write_count(self, state: _RunState) -> int:
        return sum(1 for o in state.outcomes if o.ok and getattr(self._registry.get(o.name), "mutates", False))

    def _wrote_data(self, state: _RunState) -> bool:
        return self._write_count(state) > 0

    async def _resolve_pending(self, message: str, user_id: str) -> tuple[str, list[ToolOutcome]] | None:
        """Handle a bare "yes"/"no" reply to open pending actions without calling the LLM."""
        verb = _classify_reply(message) if self._actions is not None else None
        if verb is None:
            return None
        apply = self._actions.confirm if verb == "confirm" else self._actions.cancel
        open_actions = await asyncio.to_thread(self._actions.list_open, user_id)
        if not open_actions:
            return None

        lines, outcomes = [], []
        for action in open_actions:
            t = action["payload"]
            what = f"{t['type']} of {self._currency}{t['amount']:,.2f} in '{t['category']}' on {t['date']}"
            try:
                result = await asyncio.to_thread(apply, action["id"], user_id)
            except ActionError as exc:
                error = {"error": str(exc)}
                outcomes.append(ToolOutcome(f"{verb}_action", {"action_id": action["id"]}, error, ok=False))
                lines.append(f"Couldn't {verb} the {what}: {exc}.")
                continue
            outcomes.append(ToolOutcome(f"{verb}_action", {"action_id": action["id"]}, result or {}))
            lines.append(f"Saved the {what}." if verb == "confirm" else f"Okay, discarded the {what}.")
        return "\n".join(lines), outcomes

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
                tool = self._registry.get(call.name)
                if index >= MAX_TOOL_CALLS_PER_STEP:
                    # Every tool_call id needs a reply, so refuse the excess explicitly.
                    outcome = ToolOutcome(call.name, {}, {"error": "Too many tool calls in one step"}, ok=False)
                elif tool is not None and tool.mutates and self._write_count(state) >= MAX_WRITES_PER_TURN:
                    error = {"error": f"At most {MAX_WRITES_PER_TURN} writes per message"}
                    outcome = ToolOutcome(call.name, {}, error, ok=False)
                    state.outcomes.append(outcome)
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


async def _with_deadline(events: AsyncIterator[Any], deadline: float) -> AsyncIterator[Any]:
    """Re-yield ``events``, raising ``TimeoutError`` once ``deadline`` (monotonic) passes."""
    iterator = aiter(events)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            try:
                item = await asyncio.wait_for(anext(iterator), remaining)
            except StopAsyncIteration:
                return
            yield item
    finally:
        await iterator.aclose()


def _safe_json(raw: str) -> Any:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
