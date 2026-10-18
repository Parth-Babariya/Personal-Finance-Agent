"""OpenAIChatClient against a mocked HTTP transport (no network, no API key)."""

import json

import httpx

from src.llm.client import LLMResponse, OpenAIChatClient, TextDelta


def _sse(chunks: list[dict]) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


def _chunk(delta: dict | None = None, usage: dict | None = None) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-test",
        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": None}],
        "usage": usage,
    }


def _client(chunks: list[dict], captured: list[dict]) -> OpenAIChatClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=_sse(chunks), headers={"content-type": "text/event-stream"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIChatClient("sk-test", "gpt-test", http_client=http)


async def test_stitches_fragmented_parallel_tool_calls():
    chunks = [
        _chunk({"role": "assistant", "tool_calls": [
            {"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_spending_summary", "arguments": ""}}
        ]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"period": '}}]}),
        _chunk({"tool_calls": [{"index": 1, "id": "call_b", "type": "function",
                                "function": {"name": "get_top_expenses", "arguments": "{}"}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"this_month"}'}}]}),
        _chunk(usage={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}),
    ]
    captured: list[dict] = []
    tools = [{"type": "function", "function": {"name": "x", "parameters": {"type": "object", "properties": {}}}}]

    events = [e async for e in _client(chunks, captured).stream([{"role": "user", "content": "hi"}], tools)]

    final = events[-1]
    assert isinstance(final, LLMResponse)
    assert [(c.id, c.name) for c in final.tool_calls] == [("call_a", "get_spending_summary"), ("call_b", "get_top_expenses")]
    assert json.loads(final.tool_calls[0].arguments) == {"period": "this_month"}
    assert (final.usage.prompt_tokens, final.usage.completion_tokens) == (120, 30)

    request = captured[0]
    assert request["stream"] is True and request["model"] == "gpt-test" and request["tools"] == tools


async def test_streams_text_deltas_and_omits_empty_tools():
    chunks = [_chunk({"role": "assistant", "content": "You spent "}), _chunk({"content": "₹800."})]
    captured: list[dict] = []

    events = [e async for e in _client(chunks, captured).stream([{"role": "user", "content": "q"}], [])]

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["You spent ", "₹800."]
    assert events[-1].content == "You spent ₹800." and events[-1].tool_calls == []
    assert "tools" not in captured[0]


async def test_gemini_style_calls_without_index_keep_thought_signature():
    """Gemini omits `index`, sends whole calls, and attaches a thought signature that must be echoed back."""
    signature = {"google": {"thought_signature": "sig-abc"}}
    chunks = [
        _chunk({"role": "assistant", "tool_calls": [
            {"id": "call_1", "type": "function", "extra_content": signature,
             "function": {"name": "get_spending_summary", "arguments": '{"period": "this_month"}'}}
        ]}),
        _chunk({"tool_calls": [
            {"id": "call_2", "type": "function",
             "function": {"name": "get_top_expenses", "arguments": "{}"}}
        ]}),
    ]
    events = [e async for e in _client(chunks, []).stream([{"role": "user", "content": "q"}], [])]

    calls = events[-1].tool_calls
    assert [c.id for c in calls] == ["call_1", "call_2"]  # not merged into one call
    assert calls[0].extra_content == signature and calls[1].extra_content is None
