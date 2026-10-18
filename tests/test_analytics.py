"""Trends, unusual expenses and recurring payments."""

from datetime import date, timedelta

import pytest

from src.agent.tools import ToolContext, ToolRegistry
from src.services.analytics import AnalyticsService, payee_key
from src.services.insight_generator import SimpleInsightGenerator
from tests.conftest import TODAY, fixed_today

API = "/api/v1"


@pytest.fixture
def analytics(tx_service) -> AnalyticsService:
    return AnalyticsService(tx_service, today=fixed_today)


def _add(tx_service, user_id, amount, category, description, when, type_="expense"):
    tx_service.add_transaction(
        {"amount": amount, "category": category, "description": description, "date": when, "type": type_}, user_id
    )


@pytest.fixture
def history(tx_service, user_id) -> str:
    """Four months of regular spending plus one big purchase this month."""
    for month in (date(2025, 12, 1), date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)):
        _add(tx_service, user_id, 649, "subscriptions", "Netflix subscription", month.replace(day=5))
        _add(tx_service, user_id, 15000, "rent", "House rent", month)
        _add(tx_service, user_id, 50000, "salary", "Salary", month, "income")
        for day, amount in ((3, 400), (11, 520), (19, 380), (26, 450)):
            if month.replace(day=day) <= TODAY:
                _add(tx_service, user_id, amount, "food", f"Swiggy order {day}", month.replace(day=day))
    _add(tx_service, user_id, 3200, "food", "Anniversary dinner", date(2026, 3, 14))
    return user_id


def test_payee_key_normalizes():
    assert payee_key("NETFLIX Subscription #8812") == payee_key("Netflix subscription") == "netflix subscription"


class TestTrend:
    def test_monthly_series(self, analytics, history):
        trend = analytics.monthly_trend(history, months=3)
        assert [m["month"] for m in trend["months"]] == ["2026-01", "2026-02", "2026-03"]
        jan, mar = trend["months"][0], trend["months"][2]
        assert jan["expense"] == 649 + 15000 + 400 + 520 + 380 + 450
        assert jan["income"] == 50000 and jan["partial"] is False
        assert mar["partial"] is True
        assert trend["average_monthly_expense"] == jan["expense"]  # Jan and Feb are identical

    def test_category_trend(self, analytics, history):
        months = analytics.monthly_trend(history, months=2, category="subscriptions")["months"]
        assert [m["expense"] for m in months] == [649.0, 649.0]


class TestUnusual:
    def test_flags_large_expense_in_window(self, analytics, history):
        unusual = analytics.unusual_expenses(history, TODAY - timedelta(days=29), TODAY)
        assert [u["description"] for u in unusual] == ["Anniversary dinner"]
        assert "typical food expense" in unusual[0]["reason"]

    def test_recurring_bills_are_not_unusual(self, analytics, history):
        unusual = analytics.unusual_expenses(history, date(2025, 12, 1), TODAY)
        assert "House rent" not in [u["description"] for u in unusual]

    def test_nothing_flagged_without_history(self, analytics, tx_service, user_id):
        _add(tx_service, user_id, 5000, "shopping", "Shoes", TODAY)
        assert analytics.unusual_expenses(user_id, None, None) == []


class TestRecurring:
    def test_detects_subscriptions_and_bills(self, analytics, history):
        result = analytics.recurring_payments(history)
        found = {g["description"]: g for g in result["recurring"]}
        assert set(found) == {"House rent", "Netflix subscription"}  # food orders vary too much
        assert found["Netflix subscription"]["typical_amount"] == 649.0
        assert found["Netflix subscription"]["fixed_amount"] is True
        assert found["Netflix subscription"]["next_expected"] == "2026-04-04"
        assert result["estimated_monthly_total"] == 15649.0

    def test_income_is_ignored(self, analytics, history):
        assert "Salary" not in [g["description"] for g in analytics.recurring_payments(history)["recurring"]]


async def test_tools(tx_service, analytics, history):
    registry = ToolRegistry.default(tx_service, SimpleInsightGenerator(), analytics=analytics)
    ctx = ToolContext(user_id=history, today=TODAY)
    trend = await registry.execute("get_monthly_trend", {"months": 2}, ctx)
    assert len(trend.result["months"]) == 2
    unusual = await registry.execute("find_unusual_spending", {}, ctx)
    assert unusual.result["unusual"][0]["amount"] == 3200.0
    recurring = await registry.execute("find_recurring_payments", {}, ctx)
    assert len(recurring.result["recurring"]) == 2
    bad = await registry.execute("get_monthly_trend", {"months": 40}, ctx)
    assert not bad.ok


def test_api(client, auth_headers):
    client.post(f"{API}/transactions", json={"amount": 100, "category": "food", "date": "2026-03-02"},
                headers=auth_headers)
    trend = client.get(f"{API}/analytics/trend?months=3", headers=auth_headers).json()
    assert trend["months"][-1] == {"month": "2026-03", "expense": 100.0, "income": 0.0, "net": -100.0,
                                   "partial": True}
    assert client.get(f"{API}/analytics/unusual?period=this_month", headers=auth_headers).json()["unusual"] == []
    assert client.get(f"{API}/analytics/unusual?period=bogus", headers=auth_headers).status_code == 422
    assert client.get(f"{API}/analytics/recurring", headers=auth_headers).json()["recurring"] == []
