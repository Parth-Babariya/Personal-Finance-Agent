"""Agent loop behaviour with a scripted fake LLM."""

import json

import pytest

from src.llm.client import LLMResponse, ToolCall, Usage
from tests.conftest import FakeLLM, call, make_agent, text


async def test_tool_loop_feeds_results_back_to_model(registry, conversations, seeded):
    llm = FakeLLM([
        call("get_spending_summary", period="this_month", category="food"),
        text("You spent ₹800.00 on food this month."),
    ])
    agent = make_agent(registry, conversations, llm)

    result = await agent.respond("How much on food this month?", seeded)

    assert result.mode == "llm"
    assert result.reply == "You spent ₹800.00 on food this month."
    assert result.steps == 2
    assert [c["name"] for c in result.tool_calls] == ["get_spending_summary"]
    # Second request must contain the assistant tool call and the tool result.
    second = llm.requests[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1]["role"] == "tool" and json.loads(second[-1]["content"])["total_expense"] == 800.0
    # System prompt carries the injected date and currency.
    assert "2026-03-18" in second[0]["content"] and "₹" in second[0]["content"]


async def test_parallel_tool_calls_in_one_step(registry, conversations, seeded):
    llm = FakeLLM([
        LLMResponse(tool_calls=[
            ToolCall("a", "get_spending_summary", '{"period": "this_month"}'),
            ToolCall("b", "get_spending_summary", '{"period": "last_month"}'),
        ]),
        text("done"),
    ])
    result = await make_agent(registry, conversations, llm).respond("compare", seeded)
    assert len(result.tool_calls) == 2
    tool_msgs = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"]


async def test_invalid_tool_args_are_reported_to_model(registry, conversations, seeded):
    llm = FakeLLM([call("get_spending_summary", period="fortnight"), text("Sorry, invalid period.")])
    result = await make_agent(registry, conversations, llm).respond("spend?", seeded)
    assert result.tool_calls[0]["ok"] is False
    assert "Invalid arguments" in llm.requests[1]["messages"][-1]["content"]


async def test_step_budget_forces_final_answer_without_tools(registry, conversations, seeded):
    llm = FakeLLM([call("get_insights"), call("get_insights"), text("Final answer.")])
    agent = make_agent(registry, conversations, llm, max_steps=2)
    result = await agent.respond("loop forever", seeded)
    assert result.reply == "Final answer."
    assert llm.requests[-1]["tools"] == []
    assert "budget exhausted" in llm.requests[-1]["messages"][-1]["content"]


async def test_history_is_sent_and_persisted(registry, conversations, seeded):
    llm = FakeLLM([text("first"), text("second")])
    agent = make_agent(registry, conversations, llm)
    await agent.respond("q1", seeded)
    await agent.respond("q2", seeded)

    roles_contents = [(m["role"], m.get("content")) for m in llm.requests[1]["messages"][1:]]
    assert roles_contents == [("user", "q1"), ("assistant", "first"), ("user", "q2")]
    stored = conversations.recent(seeded)
    assert [m["content"] for m in stored] == ["q1", "first", "q2", "second"]
    assert stored[-1]["meta"]["mode"] == "llm"


async def test_history_is_per_user(registry, conversations, seeded, other_user_id):
    llm = FakeLLM([text("a"), text("b")])
    agent = make_agent(registry, conversations, llm)
    await agent.respond("secret question", seeded)
    await agent.respond("hello", other_user_id)
    assert all("secret" not in (m.get("content") or "") for m in llm.requests[1]["messages"])


async def test_llm_error_falls_back_to_keyword_planner(registry, conversations, seeded):
    llm = FakeLLM([RuntimeError("provider down")])
    result = await make_agent(registry, conversations, llm).respond(
        "How much did I spend on food last month?", seeded
    )
    assert result.mode == "fallback"
    assert "₹800.00" in result.reply


async def test_no_fallback_after_a_write(registry, conversations, seeded, tx_service):
    """If the model already added a transaction, the fallback must not add it again."""
    llm = FakeLLM([call("add_transaction", amount=99, category="food"), RuntimeError("boom")])
    result = await make_agent(registry, conversations, llm).respond("I spent 99 on food", seeded)
    assert result.mode == "llm"
    assert "interrupted" in result.reply
    assert len(tx_service.list_transactions(seeded, min_amount=99, max_amount=99)) == 1


async def test_without_llm_uses_fallback(registry, conversations, seeded):
    result = await make_agent(registry, conversations, None).respond("compare this month vs last month", seeded)
    assert result.mode == "fallback"
    assert result.tool_calls[0]["name"] == "compare_periods"


async def test_stream_events_order_and_usage(registry, conversations, seeded):
    llm = FakeLLM([
        LLMResponse(tool_calls=[ToolCall("c1", "get_insights", "{}")], usage=Usage(100, 10)),
        LLMResponse(content="All good", usage=Usage(150, 5)),
    ])
    events = [e async for e in make_agent(registry, conversations, llm).stream("insights?", seeded)]
    kinds = [e["type"] for e in events]
    assert kinds[:2] == ["tool_call", "tool_result"]
    assert "token" in kinds and kinds[-1] == "done"
    assert events[-1]["usage"] == {"prompt_tokens": 250, "completion_tokens": 15}


async def test_empty_message_rejected(registry, conversations, seeded):
    with pytest.raises(ValueError):
        await make_agent(registry, conversations, None).respond("   ", seeded)


async def test_sensitive_data_is_masked_before_llm_and_storage(registry, conversations, seeded):
    llm = FakeLLM([text("Noted.")])
    result = await make_agent(registry, conversations, llm).respond(
        "My card 4111 1111 1111 1111 and PAN ABCDE1234F, how much did I spend?", seeded
    )
    sent = llm.requests[0]["messages"][-1]["content"]
    stored = conversations.recent(seeded)[0]["content"]
    for leaked in ("4111 1111 1111 1111", "ABCDE1234F"):
        assert leaked not in sent and leaked not in stored
    assert result.redacted == ["card", "pan"]


async def test_slow_llm_times_out_into_fallback(registry, conversations, seeded):
    llm = FakeLLM([text("too late")], delay=1.0)
    agent = make_agent(registry, conversations, llm, timeout_seconds=0.05)
    result = await agent.respond("How much did I spend on food last month?", seeded)
    assert result.mode == "fallback" and result.fallback_reason == "timeout"
    assert "₹800.00" in result.reply


async def test_fallback_reasons(registry, conversations, seeded):
    no_llm = await make_agent(registry, conversations, None).respond("spending this month", seeded)
    assert no_llm.fallback_reason == "no_llm"
    broken = await make_agent(registry, conversations, FakeLLM([RuntimeError("down")])).respond("spending", seeded)
    assert broken.fallback_reason == "llm_error"
    ok = await make_agent(registry, conversations, FakeLLM([text("hi")])).respond("hello", seeded)
    assert ok.mode == "llm" and ok.fallback_reason is None


async def test_done_event_returns_masked_user_message(registry, conversations, seeded):
    events = [e async for e in make_agent(registry, conversations, None).stream("PAN ABCDE1234F spend?", seeded)]
    assert events[-1]["user_message"] == "PAN [PAN] spend?"
