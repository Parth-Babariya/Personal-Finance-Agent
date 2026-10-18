"""Shared fixtures: in-memory database, fixed clock, scripted fake LLM."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.agent.agent import FinanceAgent
from src.agent.llm import LLMResponse, TextDelta, ToolCall
from src.agent.tools import ToolRegistry
from src.api.app import create_app
from src.config import Settings
from src.services.conversation_store import ConversationStore
from src.services.db import init_db, make_engine, make_session_factory
from src.services.insight_generator import SimpleInsightGenerator
from src.services.models import User
from src.services.sql_transaction_service import SQLTransactionService

TODAY = date(2026, 3, 18)  # a Wednesday


def fixed_today() -> date:
    return TODAY


class FakeLLM:
    """Replays scripted responses and records every request it receives."""

    model = "fake-model"

    def __init__(self, responses: list[LLMResponse | Exception], delay: float = 0.0):
        self._responses = list(responses)
        self._delay = delay
        self.requests: list[dict[str, Any]] = []

    async def stream(self, messages, tools) -> AsyncIterator[TextDelta | LLMResponse]:
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        response = self._responses.pop(0)
        if self._delay:
            await asyncio.sleep(self._delay)
        if isinstance(response, Exception):
            raise response
        for word in response.content.split(" ") if response.content else []:
            yield TextDelta(word + " ")
        yield response


def call(name: str, call_id: str = "c1", **arguments: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))])


def text(content: str) -> LLMResponse:
    return LLMResponse(content=content)


@pytest.fixture
def settings() -> Settings:
    return Settings(database_url="sqlite://", jwt_secret_key="test-secret", currency_symbol="₹")


@pytest.fixture
def session_factory():
    engine = make_engine("sqlite://")
    init_db(engine)
    yield make_session_factory(engine)
    engine.dispose()


@pytest.fixture
def tx_service(session_factory) -> SQLTransactionService:
    return SQLTransactionService(session_factory, today=fixed_today)


@pytest.fixture
def conversations(session_factory) -> ConversationStore:
    return ConversationStore(session_factory)


@pytest.fixture
def registry(tx_service) -> ToolRegistry:
    return ToolRegistry.default(tx_service, SimpleInsightGenerator("₹"))


def _make_user(session_factory, username: str) -> str:
    with session_factory() as session:
        user = User(username=username, password_hash="x")
        session.add(user)
        session.commit()
        return user.id


@pytest.fixture
def user_id(session_factory) -> str:
    return _make_user(session_factory, "alice")


@pytest.fixture
def other_user_id(session_factory) -> str:
    return _make_user(session_factory, "bob")


@pytest.fixture
def seeded(tx_service, user_id) -> str:
    """A small, known dataset for user_id. Returns the user id."""
    rows = [
        # this month (March 2026)
        (500.0, "expense", "food", "Swiggy dinner", date(2026, 3, 2)),
        (1200.0, "expense", "groceries", "BigBasket", date(2026, 3, 5)),
        (300.0, "expense", "food", "Lunch", date(2026, 3, 16)),
        (15000.0, "expense", "rent", "March rent", date(2026, 3, 1)),
        (60000.0, "income", "salary", "Salary", date(2026, 3, 1)),
        # last month (February 2026)
        (800.0, "expense", "food", "Restaurant", date(2026, 2, 10)),
        (15000.0, "expense", "rent", "Feb rent", date(2026, 2, 1)),
        (2500.0, "expense", "shopping", "Shoes", date(2026, 2, 20)),
    ]
    for amount, type_, category, description, when in rows:
        tx_service.add_transaction(
            {"amount": amount, "type": type_, "category": category, "description": description, "date": when},
            user_id,
        )
    return user_id


def make_agent(registry, conversations, llm=None, **kwargs) -> FinanceAgent:
    return FinanceAgent(registry, conversations, llm, currency="₹", today=fixed_today, **kwargs)


@pytest.fixture
def client(settings):
    app = create_app(settings, llm=None, today=fixed_today)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client) -> dict[str, str]:
    resp = client.post("/api/v1/auth/signup", json={"username": "tester", "password": "StrongPass123"})
    assert resp.status_code == 201
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}
