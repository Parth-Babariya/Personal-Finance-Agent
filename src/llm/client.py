"""Provider-neutral LLM client interface plus an OpenAI-compatible implementation.

Messages and tool schemas use the OpenAI chat format, which is also accepted by
Groq, Gemini (OpenAI-compatible endpoint), Ollama, vLLM and others via
``OPENAI_BASE_URL``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model
    # Provider-specific data that must be echoed back unchanged, e.g. Gemini's
    # {"google": {"thought_signature": ...}}; without it Gemini rejects the next turn.
    extra_content: dict[str, Any] | None = None


class ToolCallRejected(Exception):
    """The provider refused the model's tool call, e.g. server-side schema validation on Groq.

    Unlike an outage this is recoverable: the model can be told what was wrong and retry.
    """


def _is_tool_call_rejection(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        code = code or body.get("code") or (body.get("error") or {}).get("code")
    text = str(exc).lower()
    return code == "tool_use_failed" or "tool call validation failed" in text


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens


@dataclass
class TextDelta:
    text: str


@dataclass
class LLMResponse:
    """Final, fully assembled result of one model call."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class LLMClient(Protocol):
    model: str

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[TextDelta | LLMResponse]:
        """Yield text deltas as they arrive, then exactly one final ``LLMResponse``."""
        ...


class OpenAIChatClient:
    """Streams chat completions with tool calling from an OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float = 30.0,
        temperature: float = 0.0,
        http_client: Any = None,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self._temperature = temperature
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=2, http_client=http_client
        )

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[TextDelta | LLMResponse]:
        from openai import APIError

        try:
            async for item in self._stream(messages, tools):
                yield item
        except APIError as exc:
            if _is_tool_call_rejection(exc):
                raise ToolCallRejected(str(exc)) from exc
            raise

    async def _stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[TextDelta | LLMResponse]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        stream = await self._client.chat.completions.create(**kwargs)

        content: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage = Usage()
        async for chunk in stream:
            if chunk.usage:
                usage = Usage(chunk.usage.prompt_tokens or 0, chunk.usage.completion_tokens or 0)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content.append(delta.content)
                yield TextDelta(delta.content)
            # Tool calls arrive as fragments keyed by index; stitch them together.
            for fragment in delta.tool_calls or []:
                index = fragment.index
                if index is None:
                    # Gemini omits the index and sends each call whole: a new id starts a new call.
                    known_ids = {c["id"] for c in calls.values()}
                    index = len(calls) if fragment.id not in known_ids else max(calls)
                slot = calls.setdefault(index, {"id": "", "name": "", "arguments": "", "extra_content": None})
                if fragment.id:
                    slot["id"] = fragment.id
                if fragment.function:
                    slot["name"] += fragment.function.name or ""
                    slot["arguments"] += fragment.function.arguments or ""
                extra = getattr(fragment, "extra_content", None)
                if extra:
                    slot["extra_content"] = extra

        yield LLMResponse(
            content="".join(content),
            tool_calls=[ToolCall(**calls[i]) for i in sorted(calls)],
            usage=usage,
        )
