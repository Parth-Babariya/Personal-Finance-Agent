"""Monthly budgets: status, projections, alerts, tool and API."""

from datetime import date

import pytest

from src.services.budgets import BudgetService
from src.services.insights import SimpleInsightGenerator
from src.tools import ToolContext, ToolRegistry
from src.workflows.approvals import PendingActionService
from tests.conftest import TODAY, fixed_today, make_agent

API = "/api/v1"


@pytest.fixture
def budgets(session_factory, tx_service) -> BudgetService:
    return BudgetService(session_factory, tx_service, today=fixed_today)


class TestBudgetService:
    def test_set_update_list_delete(self, budgets, user_id):
        budgets.set_budget(user_id, "food", 3000)
        budgets.set_budget(user_id, "food", 4000)  # upsert
        budgets.set_budget(user_id, "rent", 15000)
        assert budgets.list_budgets(user_id) == {"food": 4000.0, "rent": 15000.0}
        assert budgets.delete_budget(user_id, "rent") is True
        assert budgets.delete_budget(user_id, "rent") is False

    def test_status_states_and_projection(self, budgets, seeded):
        # March so far (18 days): food 800, groceries 1200, rent 15000
        budgets.set_budget(seeded, "food", 1000)
        budgets.set_budget(seeded, "groceries", 1000)
        budgets.set_budget(seeded, "shopping", 5000)
        status = {b["category"]: b for b in budgets.status(seeded)}

        assert status["food"]["state"] == "warning" and status["food"]["used_pct"] == 80.0
        assert status["groceries"]["state"] == "over" and status["groceries"]["remaining"] == -200.0
        assert status["shopping"]["state"] == "ok" and status["shopping"]["spent"] == 0.0
        # 800 spent in 18 of 31 days -> ~1377.78 projected, over a 1000 budget
        assert status["food"]["projected_month_end"] == 1377.78
        assert status["food"]["projected_state"] == "over"

    def test_past_month_has_no_projection(self, budgets, seeded):
        budgets.set_budget(seeded, "shopping", 2000)
        [feb] = budgets.status(seeded, month=date(2026, 2, 1))
        assert feb["spent"] == 2500.0 and feb["state"] == "over"
        assert "projected_month_end" not in feb

    def test_impact_and_alert(self, budgets, seeded):
        budgets.set_budget(seeded, "food", 1000)
        impact = budgets.impact(seeded, "food", 300, TODAY)
        assert impact["spent_after"] == 1100.0 and impact["state_after"] == "over"
        assert budgets.impact(seeded, "travel", 300, TODAY) is None
        assert "used 80% of your food budget" in budgets.alert_for(seeded, "food", TODAY)
        assert budgets.alert_for(seeded, "travel", TODAY) is None

    def test_budgets_are_per_user(self, budgets, user_id, other_user_id):
        budgets.set_budget(user_id, "food", 1000)
        assert budgets.list_budgets(other_user_id) == {}


class TestBudgetTool:
    async def test_check_budget_tool(self, tx_service, budgets, seeded):
        registry = ToolRegistry.default(tx_service, SimpleInsightGenerator(), budgets=budgets)
        ctx = ToolContext(user_id=seeded, today=TODAY)
        none_set = await registry.execute("check_budget", {}, ctx)
        assert none_set.result["budgets"] == [] and "not set a budget" in none_set.result["note"]

        budgets.set_budget(seeded, "food", 1000)
        out = await registry.execute("check_budget", {"category": "food"}, ctx)
        assert out.result["budgets"][0]["state"] == "warning"

    def test_tool_only_registered_with_service(self, tx_service, budgets):
        assert "check_budget" not in ToolRegistry.default(tx_service, SimpleInsightGenerator()).names
        assert "check_budget" in ToolRegistry.default(tx_service, SimpleInsightGenerator(), budgets=budgets).names

    async def test_proposal_includes_budget_impact_and_confirm_reports_alert(
        self, session_factory, tx_service, budgets, conversations, seeded
    ):
        budgets.set_budget(seeded, "food", 1000)
        actions = PendingActionService(session_factory, tx_service, budgets=budgets)
        registry = ToolRegistry.default(tx_service, SimpleInsightGenerator(), actions, budgets=budgets)
        ctx = ToolContext(user_id=seeded, today=TODAY)

        proposal = await registry.execute("add_transaction", {"amount": 300, "category": "food"}, ctx)
        assert proposal.result["budget_impact"]["state_after"] == "over"

        agent = make_agent(registry, conversations, None, actions=actions)
        confirmation = await agent.respond("yes", seeded)
        assert "over your food budget" in confirmation.reply


class TestBudgetAPI:
    def test_crud_status_and_alert_on_add(self, client, auth_headers):
        assert client.put(f"{API}/budgets/food", json={"monthly_limit": 500}, headers=auth_headers).status_code == 200
        assert client.put(f"{API}/budgets/salary", json={"monthly_limit": 1}, headers=auth_headers).status_code == 422
        assert client.get(f"{API}/budgets", headers=auth_headers).json() == {"food": 500.0}

        quiet = client.post(f"{API}/transactions", json={"amount": 100, "category": "food"}, headers=auth_headers)
        assert quiet.json()["budget_alert"] is None
        loud = client.post(f"{API}/transactions", json={"amount": 350, "category": "food"}, headers=auth_headers)
        assert "90%" in loud.json()["budget_alert"]

        [status] = client.get(f"{API}/budgets/status", headers=auth_headers).json()
        assert status["spent"] == 450.0 and status["state"] == "warning"

        assert client.delete(f"{API}/budgets/food", headers=auth_headers).status_code == 204
        assert client.delete(f"{API}/budgets/food", headers=auth_headers).status_code == 404
