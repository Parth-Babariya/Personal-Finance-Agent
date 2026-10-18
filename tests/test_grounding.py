"""Runtime grounding check for money figures in LLM replies."""

import pytest

from src.agent.grounding import check_grounding, extract_money
from tests.conftest import FakeLLM, call, make_agent, text

SUMMARY = {"total_expense": 17000.0, "total_income": 60000.0, "net": 43000.0, "by_category": {"food": 800.0}}


@pytest.mark.parametrize(
    ("snippet", "value"),
    [("₹1,234.50", 1234.5), ("Rs. 450", 450.0), ("INR 2,000", 2000.0), ("$12.5", 12.5), ("₹45k", 45000.0),
     ("₹1.2 lakh", 120000.0), ("₹2cr", 20000000.0)],
)
def test_extract_money(snippet, value):
    assert extract_money(f"you spent {snippet} total", "₹")[0][0] == value


def test_ignores_non_money_numbers():
    assert extract_money("22 transactions in 2026, 72% saved", "₹") == []


@pytest.mark.parametrize(
    "reply",
    [
        "You spent ₹17,000.00 and earned ₹60,000.00.",
        "Net savings were ₹43,000.",  # rounded
        "Food cost ₹800, which is ₹16,200 less than your total.",  # derived difference
        "You've spent about ₹17k this month.",  # abbreviated
        "No figures here.",
    ],
)
def test_grounded_replies_pass(reply):
    assert check_grounding(reply, [SUMMARY], [], "₹").verified


def test_invented_figure_is_flagged():
    report = check_grounding("You spent ₹17,000 and ₹2,345 on travel.", [SUMMARY], [], "₹")
    assert not report.verified and report.unverified == [2345.0]


def test_numbers_from_user_message_are_allowed():
    report = check_grounding("Got it, ₹450 for lunch.", [], ["I spent 450 on lunch"], "₹")
    assert report.verified


async def test_agent_retries_once_when_reply_is_ungrounded(registry, conversations, seeded):
    llm = FakeLLM([
        call("get_spending_summary", period="this_month", category="food"),
        text("You spent ₹999.00 on food."),  # invented
        text("You spent ₹800.00 on food."),  # corrected
    ])
    events = [e async for e in make_agent(registry, conversations, llm).stream("food this month?", seeded)]
    done = events[-1]

    assert "reset" in [e["type"] for e in events]
    assert done["reply"] == "You spent ₹800.00 on food."
    assert done["grounding"] == {"checked": 1, "verified": True, "unverified": [], "retried": True}
    assert "₹999.00" in llm.requests[2]["messages"][-1]["content"]  # correction names the bad figure


async def test_agent_flags_reply_that_stays_ungrounded(registry, conversations, seeded):
    llm = FakeLLM([
        call("get_spending_summary", period="this_month"),
        text("You spent ₹12,345.00."),
        text("You spent ₹12,345.00."),
    ])
    result = await make_agent(registry, conversations, llm).respond("spend?", seeded)
    assert result.grounding["verified"] is False and result.grounding["unverified"] == [12345.0]
    assert "couldn't verify ₹12,345.00" in result.reply


async def test_grounded_reply_is_not_retried(registry, conversations, seeded):
    llm = FakeLLM([call("get_spending_summary", period="this_month", category="food"), text("₹800.00 on food.")])
    result = await make_agent(registry, conversations, llm).respond("food?", seeded)
    assert result.grounding == {"checked": 1, "verified": True, "unverified": [], "retried": False}
    assert len(llm.requests) == 2


def test_numbers_quoted_inside_tool_text_are_grounded():
    insights = {"insights": ["Your highest spending category is 'shopping' at ₹6,042.00 (36% of expenses)."]}
    unusual = {"unusual": [{"amount": 12000.0, "reason": "4.4x your typical shopping expense (median 2,700.00)"}]}
    assert check_grounding("Shopping ₹6,042; median ₹2,700.", [insights, unusual], [], "₹").verified
