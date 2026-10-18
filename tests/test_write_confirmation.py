"""Agent writes are proposed first and only applied after explicit confirmation."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.llm.client import LLMResponse, ToolCall
from src.services.insights import SimpleInsightGenerator
from src.tools import ToolContext, ToolRegistry
from src.workflows.approvals import ActionError, PendingActionService
from tests.conftest import TODAY, FakeLLM, call, fixed_today, make_agent, text


class Clock:
    def __init__(self):
        self.now = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def actions(session_factory, tx_service, clock) -> PendingActionService:
    return PendingActionService(session_factory, tx_service, ttl_minutes=15, now=clock)


@pytest.fixture
def confirm_registry(tx_service, actions) -> ToolRegistry:
    return ToolRegistry.default(tx_service, SimpleInsightGenerator("₹"), actions)


def _expense(amount: float = 450.0, category: str = "food") -> dict:
    return {"amount": amount, "type": "expense", "category": category, "description": "lunch",
            "date": TODAY.isoformat()}


class TestPendingActionService:
    def test_confirm_applies_write_once(self, actions, tx_service, user_id):
        action = actions.propose(user_id, "add_transaction", _expense())
        assert tx_service.list_transactions(user_id) == []
        confirmed = actions.confirm(action["id"], user_id)
        assert confirmed["status"] == "confirmed" and confirmed["result"]["added"]["amount"] == 450.0
        assert len(tx_service.list_transactions(user_id)) == 1
        with pytest.raises(ActionError, match="already confirmed"):
            actions.confirm(action["id"], user_id)

    def test_cancel_and_expiry(self, actions, tx_service, user_id, clock):
        cancelled = actions.propose(user_id, "add_transaction", _expense())
        assert actions.cancel(cancelled["id"], user_id)["status"] == "cancelled"

        stale = actions.propose(user_id, "add_transaction", _expense(99))
        clock.now += timedelta(minutes=16)
        assert actions.list_open(user_id) == []
        with pytest.raises(ActionError, match="expired"):
            actions.confirm(stale["id"], user_id)
        assert tx_service.list_transactions(user_id) == []

    def test_actions_are_user_scoped(self, actions, user_id, other_user_id):
        action = actions.propose(user_id, "add_transaction", _expense())
        assert actions.confirm(action["id"], other_user_id) is None
        assert actions.cancel(action["id"], other_user_id) is None
        assert actions.list_open(other_user_id) == []


async def test_tool_proposes_instead_of_writing(confirm_registry, tx_service, seeded):
    ctx = ToolContext(user_id=seeded, today=TODAY)
    out = await confirm_registry.execute("add_transaction", {"amount": 450, "category": "food"}, ctx)
    assert out.ok and out.result["status"] == "pending_confirmation"
    assert "action_id" in out.result and "Not saved yet" in out.result["instructions"]
    assert tx_service.list_transactions(seeded, min_amount=450, max_amount=450) == []


async def test_tool_flags_possible_duplicates(confirm_registry, tx_service, seeded):
    ctx = ToolContext(user_id=seeded, today=TODAY)
    tx_service.add_transaction({"amount": 450, "category": "food", "date": TODAY}, seeded)
    out = await confirm_registry.execute("add_transaction", {"amount": 450, "category": "food"}, ctx)
    assert len(out.result["possible_duplicates"]) == 1


class TestAgentConfirmation:
    async def test_yes_confirms_without_calling_llm(self, confirm_registry, conversations, actions, tx_service,
                                                    seeded):
        llm = FakeLLM([call("add_transaction", amount=450, category="food"),
                       text("I'll record ₹450.00 for food today. Confirm?")])
        agent = make_agent(confirm_registry, conversations, llm, actions=actions)

        proposal = await agent.respond("I spent 450 on lunch", seeded)
        assert len(proposal.pending_actions) == 1
        assert tx_service.list_transactions(seeded, min_amount=450, max_amount=450) == []

        confirmation = await agent.respond("yes", seeded)  # FakeLLM has no responses left
        assert confirmation.mode == "action"
        assert "Saved the expense of ₹450.00" in confirmation.reply
        assert len(tx_service.list_transactions(seeded, min_amount=450, max_amount=450)) == 1
        assert len(llm.requests) == 2

    async def test_no_discards(self, confirm_registry, conversations, actions, tx_service, seeded):
        llm = FakeLLM([call("add_transaction", amount=450, category="food"), text("Confirm?")])
        agent = make_agent(confirm_registry, conversations, llm, actions=actions)
        await agent.respond("I spent 450 on lunch", seeded)
        result = await agent.respond("no, cancel that", seeded)
        assert result.mode == "action" and "discarded" in result.reply
        assert actions.list_open(seeded) == []
        assert tx_service.list_transactions(seeded, min_amount=450, max_amount=450) == []

    async def test_yes_without_open_action_goes_to_llm(self, confirm_registry, conversations, actions, seeded):
        llm = FakeLLM([text("Yes to what?")])
        result = await make_agent(confirm_registry, conversations, llm, actions=actions).respond("yes", seeded)
        assert result.mode == "llm"

    async def test_fallback_proposes_then_confirms(self, confirm_registry, conversations, actions, tx_service,
                                                   seeded):
        agent = make_agent(confirm_registry, conversations, None, actions=actions)
        proposal = await agent.respond("I spent 450 on lunch", seeded)
        assert "Reply 'yes' to confirm" in proposal.reply
        await agent.respond("yes", seeded)
        assert len(tx_service.list_transactions(seeded, min_amount=450, max_amount=450)) == 1

    async def test_write_cap_per_message(self, confirm_registry, conversations, actions, seeded):
        calls = [ToolCall(f"c{i}", "add_transaction", f'{{"amount": {10 + i}}}') for i in range(4)]
        llm = FakeLLM([LLMResponse(tool_calls=calls), text("done")])
        result = await make_agent(confirm_registry, conversations, llm, actions=actions).respond("log 4", seeded)
        assert [c["ok"] for c in result.tool_calls] == [True, True, True, False]
        assert "writes per message" in result.tool_calls[-1]["result"]["error"]
        assert len(actions.list_open(seeded)) == 3


class TestActionsAPI:
    def test_confirm_via_api(self, settings):
        llm = FakeLLM([call("add_transaction", amount=250, category="transport"), text("Confirm?")])
        app = create_app(settings, llm=llm, today=fixed_today)
        with TestClient(app) as client:
            token = client.post("/api/v1/auth/signup", json={"username": "conf", "password": "StrongPass123"})
            headers = {"Authorization": f"Bearer {token.json()['access_token']}"}

            chat = client.post("/api/v1/chat", json={"message": "uber 250"}, headers=headers).json()
            [pending] = chat["pending_actions"]
            assert pending["payload"]["amount"] == 250.0
            assert [a["id"] for a in client.get("/api/v1/actions", headers=headers).json()] == [pending["id"]]

            other = client.post("/api/v1/auth/signup", json={"username": "other", "password": "StrongPass123"})
            other_headers = {"Authorization": f"Bearer {other.json()['access_token']}"}
            assert client.post(f"/api/v1/actions/{pending['id']}/confirm", headers=other_headers).status_code == 404

            resp = client.post(f"/api/v1/actions/{pending['id']}/confirm", headers=headers)
            assert resp.status_code == 200 and resp.json()["status"] == "confirmed"
            assert client.post(f"/api/v1/actions/{pending['id']}/confirm", headers=headers).status_code == 409
            assert len(client.get("/api/v1/transactions", headers=headers).json()) == 1


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("yes", "confirm"), ("Yes please!", "confirm"), ("ok, save it", "confirm"), ("go ahead", "confirm"),
        ("no", "cancel"), ("no, cancel that", "cancel"), ("don't save it", "cancel"), ("Nope.", "cancel"),
        ("yes, how much did I spend?", None), ("I spent 450 on lunch", None), ("it", None), ("", None),
    ],
)
def test_classify_reply(message, expected):
    from src.agents.finance_agent import _classify_reply

    assert _classify_reply(message) == expected
