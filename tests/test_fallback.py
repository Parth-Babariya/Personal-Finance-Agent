"""Deterministic keyword planner used without an LLM."""

import pytest

from src.agent.fallback import HELP_TEXT, KeywordFallbackPlanner
from src.agent.tools import ToolContext
from tests.conftest import TODAY


@pytest.fixture
def planner(registry) -> KeywordFallbackPlanner:
    return KeywordFallbackPlanner(registry, "₹")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("How much did I spend on food last month?",
         [("get_spending_summary", {"period": "last_month", "category": "food"})]),
        ("What's my spending this week",
         [("get_spending_summary", {"period": "this_week"}), ("get_insights", {"period": "this_week"})]),
        ("Compare this month vs last month",
         [("compare_periods", {"period_a": "this_month", "period_b": "last_month"})]),
        ("Show my biggest expenses in the last 30 days",
         [("get_top_expenses", {"period": "last_30_days"})]),
        ("list my uber transactions",
         [("search_transactions", {"period": "all_time", "limit": 10, "category": "transport"})]),
        ("any tips on my habits?", [("get_insights", {"period": "this_month"})]),
        ("hello there", []),
    ],
)
def test_plan(planner, message, expected):
    assert planner.plan(message) == expected


def test_plan_add_expense_and_income(planner):
    [(name, args)] = planner.plan("I spent ₹1,250 on groceries at BigBasket")
    assert name == "add_transaction"
    assert args["amount"] == 1250.0 and args["category"] == "groceries" and args["type"] == "expense"

    [(name, args)] = planner.plan("received 60000 salary")
    assert args["type"] == "income" and args["category"] == "salary"


def test_follow_up_inherits_previous_question(planner):
    history = [
        {"role": "user", "content": "How much did I spend on food this month?"},
        {"role": "assistant", "content": "..."},
    ]
    assert planner.plan("what about last month?", history) == [
        ("get_spending_summary", {"period": "last_month", "category": "food"})
    ]


async def test_respond_renders_numbers(planner, seeded):
    ctx = ToolContext(user_id=seeded, today=TODAY)
    reply, outcomes = await planner.respond("How much did I spend on food last month?", ctx)
    assert [o.name for o in outcomes] == ["get_spending_summary"]
    assert "₹800.00" in reply

    reply, _ = await planner.respond("compare this month with last month", ctx)
    assert "₹17,000.00" in reply and "₹18,300.00" in reply and "less" in reply


async def test_respond_help_text_without_tools(planner, seeded):
    reply, outcomes = await planner.respond("hi", ToolContext(user_id=seeded, today=TODAY))
    assert reply == HELP_TEXT and outcomes == []
