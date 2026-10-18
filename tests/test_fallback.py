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


@pytest.fixture
def full_planner(session_factory, tx_service):
    from src.agent.tools import ToolRegistry
    from src.services.analytics import AnalyticsService
    from src.services.budgets import BudgetService
    from src.services.goals import GoalService
    from src.services.insight_generator import SimpleInsightGenerator
    from tests.conftest import fixed_today

    budgets = BudgetService(session_factory, tx_service, today=fixed_today)
    goals = GoalService(session_factory, tx_service, today=fixed_today)
    registry = ToolRegistry.default(
        tx_service, SimpleInsightGenerator("₹"), budgets=budgets,
        analytics=AnalyticsService(tx_service, today=fixed_today), goals=goals,
    )
    return KeywordFallbackPlanner(registry, "₹"), budgets, goals


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Am I within my food budget?", [("check_budget", {"category": "food"})]),
        ("What subscriptions am I paying for?", [("find_recurring_payments", {})]),
        ("How much did I spend on subscriptions this month?",
         [("get_spending_summary", {"period": "this_month", "category": "subscriptions"})]),
        ("Anything unusual this month?", [("find_unusual_spending", {"period": "this_month"})]),
        ("Show my spending trend", [("get_monthly_trend", {"months": 6})]),
        ("How are my savings goals going?", [("get_savings_goals", {})]),
    ],
)
def test_plan_feature_tools(full_planner, message, expected):
    planner, _, _ = full_planner
    assert planner.plan(message) == expected


def test_feature_tools_not_planned_when_unavailable(planner):
    assert planner.plan("Am I within my food budget?")[0][0] == "get_spending_summary"


async def test_render_feature_tools(full_planner, seeded):
    planner, budgets, goals = full_planner
    ctx = ToolContext(user_id=seeded, today=TODAY)

    reply, _ = await planner.respond("how is my budget?", ctx)
    assert "not set a budget" in reply
    budgets.set_budget(seeded, "food", 1000)
    reply, _ = await planner.respond("how is my budget?", ctx)
    assert "food: ₹800.00 of ₹1,000.00 (80%, warning)" in reply and "on pace for" in reply

    reply, _ = await planner.respond("show my monthly trend", ctx)
    assert "2026-03 ₹17,000.00 (so far)" in reply and "2026-02 ₹18,300.00" in reply

    goals.create(seeded, "Goa trip", 50000, saved_amount=10000)
    reply, _ = await planner.respond("my goals?", ctx)
    assert "Goa trip: ₹10,000.00 of ₹50,000.00 (20%)" in reply

    reply, _ = await planner.respond("any recurring bills?", ctx)
    assert "didn't find any recurring payments" in reply
