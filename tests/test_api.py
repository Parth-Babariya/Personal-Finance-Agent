"""HTTP-level tests against an app built with an in-memory DB and no LLM."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.core.config import Settings
from src.core.security import create_access_token
from tests.conftest import FakeLLM, call, fixed_today, text

API = "/api/v1"


def _add(client, headers, **body):
    payload = {"amount": 100.0, "category": "food", **body}
    resp = client.post(f"{API}/transactions", json=payload, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


class TestAuth:
    def test_signup_login_and_duplicate(self, client):
        creds = {"username": "neo", "password": "StrongPass123"}
        assert client.post(f"{API}/auth/signup", json=creds).status_code == 201
        assert client.post(f"{API}/auth/signup", json=creds).status_code == 409
        assert "access_token" in client.post(f"{API}/auth/login", json=creds).json()
        bad = {**creds, "password": "WrongPass123"}
        assert client.post(f"{API}/auth/login", json=bad).status_code == 401

    def test_oauth2_form_token(self, client, auth_headers):
        resp = client.post(f"{API}/auth/token", data={"username": "tester", "password": "StrongPass123"})
        assert resp.status_code == 200

    @pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer not.a.token"}])
    def test_protected_routes_reject_bad_tokens(self, client, headers):
        assert client.get(f"{API}/transactions", headers=headers).status_code == 401

    def test_token_for_unknown_user_or_wrong_secret(self, client, settings):
        ghost = create_access_token("ghost", settings.jwt_secret_key)
        forged = create_access_token("ghost", "other-secret")
        for token in (ghost, forged):
            resp = client.get(f"{API}/transactions", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 401


class TestTransactions:
    def test_crud_roundtrip(self, client, auth_headers):
        tx = _add(client, auth_headers, amount=42.5, description="Lunch", date="2026-03-10")
        assert tx["date"] == "2026-03-10" and tx["type"] == "expense"

        resp = client.patch(f"{API}/transactions/{tx['id']}", json={"amount": 50}, headers=auth_headers)
        assert resp.json()["amount"] == 50.0 and resp.json()["description"] == "Lunch"

        assert client.get(f"{API}/transactions/{tx['id']}", headers=auth_headers).status_code == 200
        assert client.delete(f"{API}/transactions/{tx['id']}", headers=auth_headers).status_code == 204
        assert client.get(f"{API}/transactions/{tx['id']}", headers=auth_headers).status_code == 404

    @pytest.mark.parametrize(
        "body",
        [
            {"amount": -5},
            {"amount": 0},
            {"amount": 10, "category": "crypto"},
            {"amount": 10, "type": "transfer"},
            {"amount": 10, "user_id": "someone"},
        ],
    )
    def test_validation(self, client, auth_headers, body):
        assert client.post(f"{API}/transactions", json=body, headers=auth_headers).status_code == 422

    def test_filters_and_summary_period(self, client, auth_headers):
        _add(client, auth_headers, amount=100, date="2026-03-02")
        _add(client, auth_headers, amount=300, category="rent", date="2026-02-15")
        _add(client, auth_headers, amount=1000, type="income", category="salary", date="2026-03-01")

        march = client.get(f"{API}/transactions?period=this_month", headers=auth_headers).json()
        assert len(march) == 2
        rent = client.get(f"{API}/transactions?category=rent", headers=auth_headers).json()
        assert [t["amount"] for t in rent] == [300.0]

        s = client.get(f"{API}/summary?period=month", headers=auth_headers).json()
        assert s["period"] == "this_month"
        assert (s["total_expense"], s["total_income"], s["net"]) == (100.0, 1000.0, 900.0)

        feb = client.get(f"{API}/summary?start_date=2026-02-01&end_date=2026-02-28", headers=auth_headers).json()
        assert feb["total_expense"] == 300.0

        assert client.get(f"{API}/summary?period=fortnight", headers=auth_headers).status_code == 422
        bad_range = f"{API}/summary?start_date=2026-03-10&end_date=2026-03-01"
        assert client.get(bad_range, headers=auth_headers).status_code == 422

    def test_users_cannot_touch_each_others_data(self, client, auth_headers):
        tx = _add(client, auth_headers)
        other = client.post(f"{API}/auth/signup", json={"username": "mallory", "password": "StrongPass123"})
        other_headers = {"Authorization": f"Bearer {other.json()['access_token']}"}
        assert client.get(f"{API}/transactions", headers=other_headers).json() == []
        assert client.get(f"{API}/transactions/{tx['id']}", headers=other_headers).status_code == 404
        assert client.delete(f"{API}/transactions/{tx['id']}", headers=other_headers).status_code == 404


class TestChat:
    def test_chat_fallback_and_history(self, client, auth_headers):
        _add(client, auth_headers, amount=250, category="food", date="2026-03-05")
        resp = client.post(f"{API}/chat", json={"message": "How much did I spend on food this month?"},
                           headers=auth_headers)
        body = resp.json()
        assert resp.status_code == 200
        assert body["mode"] == "fallback" and "₹250.00" in body["reply"]
        assert body["tool_calls"][0]["name"] == "get_spending_summary"

        history = client.get(f"{API}/chat/history", headers=auth_headers).json()
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert len(client.get(f"{API}/memory/search?query=food", headers=auth_headers).json()) == 2
        assert client.delete(f"{API}/chat/history", headers=auth_headers).json() == {"deleted": 2}

    def test_chat_validation(self, client, auth_headers):
        assert client.post(f"{API}/chat", json={"message": ""}, headers=auth_headers).status_code == 422
        assert client.post(f"{API}/chat", json={"message": "x" * 2001}, headers=auth_headers).status_code == 422

    def test_chat_stream_sse(self, settings):
        llm = FakeLLM([call("get_spending_summary", period="this_month"), text("You spent ₹0.00.")])
        app = create_app(settings, llm=llm, today=fixed_today)
        with TestClient(app) as client:
            token = client.post(f"{API}/auth/signup", json={"username": "streamer", "password": "StrongPass123"})
            headers = {"Authorization": f"Bearer {token.json()['access_token']}"}
            with client.stream("POST", f"{API}/chat/stream", json={"message": "spend?"}, headers=headers) as resp:
                assert resp.headers["content-type"].startswith("text/event-stream")
                events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
        kinds = [e["type"] for e in events]
        assert kinds[0] == "tool_call" and kinds[-1] == "done"
        assert events[-1]["reply"] == "You spent ₹0.00." and events[-1]["mode"] == "llm"


def test_health_reports_llm_status(client):
    assert client.get("/health").json() == {"status": "ok", "llm_enabled": False}


def test_production_requires_jwt_secret(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        Settings.from_env()


def test_dev_generates_ephemeral_secret(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    assert len(Settings.from_env().jwt_secret_key) > 32
