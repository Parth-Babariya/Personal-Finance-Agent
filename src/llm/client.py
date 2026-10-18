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
        calls: dict[int, dict[str, str]] = {}
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
                slot = calls.setdefault(fragment.index, {"id": "", "name": "", "arguments": ""})
                if fragment.id:
                    slot["id"] = fragment.id
                if fragment.function:
                    slot["name"] += fragment.function.name or ""
                    slot["arguments"] += fragment.function.arguments or ""

        yield LLMResponse(
            content="".join(content),
            tool_calls=[ToolCall(**calls[i]) for i in sorted(calls)],
            usage=usage,
        )
