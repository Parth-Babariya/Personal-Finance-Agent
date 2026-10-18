"""Rate limiting, login lockout and the daily LLM token budget."""

from dataclasses import replace

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.guardrails.rate_limit import SlidingWindowLimiter
from src.llm.client import LLMResponse, Usage
from src.services.usage import UsageService
from tests.conftest import FakeLLM, fixed_today, make_agent

API = "/api/v1"


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class TestSlidingWindowLimiter:
    def test_limit_and_window(self):
        clock = FakeClock()
        limiter = SlidingWindowLimiter(2, 60, clock=clock)
        assert limiter.hit("u") is None and limiter.hit("u") is None
        assert limiter.hit("u") == 60.0
        assert limiter.hit("other") is None  # keys are independent
        clock.t += 61
        assert limiter.hit("u") is None

    def test_retry_after_does_not_consume_and_reset_clears(self):
        limiter = SlidingWindowLimiter(1, 60, clock=FakeClock())
        assert limiter.retry_after("u") is None
        limiter.hit("u")
        assert limiter.retry_after("u") == 60.0
        limiter.reset("u")
        assert limiter.retry_after("u") is None

    def test_zero_disables(self):
        limiter = SlidingWindowLimiter(0, 60)
        assert all(limiter.hit("u") is None for _ in range(100))


def _client(settings, **overrides) -> TestClient:
    return TestClient(create_app(replace(settings, **overrides), llm=None, today=fixed_today))


def test_chat_rate_limit_returns_429_with_retry_after(settings):
    with _client(settings, chat_requests_per_minute=2) as client:
        token = client.post(f"{API}/auth/signup", json={"username": "chatty", "password": "StrongPass123"})
        headers = {"Authorization": f"Bearer {token.json()['access_token']}"}
        codes = [client.post(f"{API}/chat", json={"message": "hi"}, headers=headers).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        resp = client.post(f"{API}/chat", json={"message": "hi"}, headers=headers)
        assert int(resp.headers["Retry-After"]) >= 1


def test_auth_endpoints_rate_limited_per_ip(settings):
    with _client(settings, auth_requests_per_minute=3) as client:
        creds = {"username": "someone", "password": "WrongPass123"}
        codes = [client.post(f"{API}/auth/login", json=creds).status_code for _ in range(4)]
        assert codes == [401, 401, 401, 429]


def test_login_lockout_after_repeated_failures(settings):
    with _client(settings, login_max_failures=3, auth_requests_per_minute=0) as client:
        creds = {"username": "victim", "password": "StrongPass123"}
        client.post(f"{API}/auth/signup", json=creds)
        bad = {**creds, "password": "WrongPass123"}
        assert [client.post(f"{API}/auth/login", json=bad).status_code for _ in range(3)] == [401, 401, 401]
        locked = client.post(f"{API}/auth/login", json=creds)  # even the right password is refused now
        assert locked.status_code == 429 and "failed login" in locked.json()["detail"]


def test_successful_login_resets_failure_count(settings):
    with _client(settings, login_max_failures=2, auth_requests_per_minute=0) as client:
        creds = {"username": "forgetful", "password": "StrongPass123"}
        client.post(f"{API}/auth/signup", json=creds)
        bad = {**creds, "password": "WrongPass123"}
        client.post(f"{API}/auth/login", json=bad)
        assert client.post(f"{API}/auth/login", json=creds).status_code == 200
        assert client.post(f"{API}/auth/login", json=bad).status_code == 401  # counter was reset


class TestTokenBudget:
    async def test_usage_is_recorded_per_day(self, session_factory, registry, conversations, seeded):
        usage = UsageService(session_factory, today=fixed_today)
        llm = FakeLLM([LLMResponse(content="hello", usage=Usage(120, 30))])
        await make_agent(registry, conversations, llm, usage=usage, daily_token_budget=1000).respond("hi", seeded)
        assert usage.tokens_today(seeded) == 150

    async def test_over_budget_switches_to_fallback(self, session_factory, registry, conversations, seeded):
        usage = UsageService(session_factory, today=fixed_today)
        usage.record(seeded, 900, 200)
        llm = FakeLLM([])  # must not be called
        agent = make_agent(registry, conversations, llm, usage=usage, daily_token_budget=1000)
        result = await agent.respond("How much did I spend on food last month?", seeded)
        assert result.mode == "fallback" and result.fallback_reason == "budget"
        assert "₹800.00" in result.reply and llm.requests == []

    async def test_budget_zero_means_unlimited(self, session_factory, registry, conversations, seeded):
        usage = UsageService(session_factory, today=fixed_today)
        usage.record(seeded, 10**9, 0)
        llm = FakeLLM([LLMResponse(content="still here")])
        result = await make_agent(registry, conversations, llm, usage=usage, daily_token_budget=0).respond("hi", seeded)
        assert result.mode == "llm"
