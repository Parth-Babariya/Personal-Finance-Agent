"""Savings goals: progress, saving-rate projection, tool and API."""

from datetime import date

import pytest

from src.agent.tools import ToolContext, ToolRegistry
from src.services.goals import GoalService
from src.services.insight_generator import SimpleInsightGenerator
from tests.conftest import TODAY, fixed_today

API = "/api/v1"


@pytest.fixture
def goals(session_factory, tx_service) -> GoalService:
    return GoalService(session_factory, tx_service, today=fixed_today)


def test_average_monthly_saving_uses_complete_months(goals, seeded):
    # Only February is a complete month with data in the seeded set: income 0 - expenses 18300
    assert goals.average_monthly_saving(seeded) == -18300.0


def test_progress_and_on_track(goals, tx_service, user_id):
    for month in (date(2025, 12, 1), date(2026, 1, 1), date(2026, 2, 1)):
        tx_service.add_transaction({"amount": 50000, "type": "income", "category": "salary", "date": month}, user_id)
        tx_service.add_transaction({"amount": 40000, "category": "rent", "date": month}, user_id)

    goal = goals.create(user_id, "Goa trip", 50000, date(2026, 9, 1), saved_amount=20000)
    assert goal["progress_pct"] == 40.0 and goal["remaining"] == 30000.0
    assert goal["months_left"] == 6 and goal["required_per_month"] == 5000.0
    assert goal["average_monthly_saving"] == 10000.0 and goal["on_track"] is True
    assert goal["projected_completion"] == "2026-06"

    stretched = goals.create(user_id, "Car", 500000, date(2026, 12, 1))
    assert stretched["on_track"] is False


def test_contribute_complete_and_withdraw_limits(goals, user_id):
    goal = goals.create(user_id, "Laptop", 1000)
    done = goals.contribute(goal["id"], user_id, 1000)
    assert done["completed"] is True and done["progress_pct"] == 100.0
    with pytest.raises(ValueError):
        goals.contribute(goal["id"], user_id, -5000)


def test_validation_and_scoping(goals, user_id, other_user_id):
    with pytest.raises(ValueError):
        goals.create(user_id, "Past", 100, TODAY)
    goal = goals.create(user_id, "Mine", 100)
    assert goals.contribute(goal["id"], other_user_id, 10) is None
    assert goals.delete(goal["id"], other_user_id) is False
    assert goals.list(other_user_id) == []
    assert goals.delete(goal["id"], user_id) is True


async def test_tool(tx_service, goals, user_id):
    registry = ToolRegistry.default(tx_service, SimpleInsightGenerator(), goals=goals)
    ctx = ToolContext(user_id=user_id, today=TODAY)
    empty = await registry.execute("get_savings_goals", {}, ctx)
    assert empty.result["goals"] == [] and "no savings goals" in empty.result["note"]
    goals.create(user_id, "Goa trip", 50000)
    out = await registry.execute("get_savings_goals", {}, ctx)
    assert out.result["goals"][0]["name"] == "Goa trip"


def test_api(client, auth_headers):
    resp = client.post(f"{API}/goals", json={"name": "Goa trip", "target_amount": 50000, "target_date": "2026-09-01"},
                       headers=auth_headers)
    assert resp.status_code == 201
    goal_id = resp.json()["id"]
    assert client.post(f"{API}/goals", json={"name": "x", "target_amount": 10, "target_date": "2020-01-01"},
                       headers=auth_headers).status_code == 422
    updated = client.post(f"{API}/goals/{goal_id}/contributions", json={"amount": 5000}, headers=auth_headers)
    assert updated.json()["saved_amount"] == 5000.0
    assert [g["id"] for g in client.get(f"{API}/goals", headers=auth_headers).json()] == [goal_id]
    assert client.delete(f"{API}/goals/{goal_id}", headers=auth_headers).status_code == 204
    assert client.post(f"{API}/goals/{goal_id}/contributions", json={"amount": 1},
                       headers=auth_headers).status_code == 404
