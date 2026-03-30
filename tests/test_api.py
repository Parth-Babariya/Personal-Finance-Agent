"""Tests for the Personal Finance Agent API.

These tests verify the API endpoints work correctly with the in-memory
service implementations. The TestClient is used as a context manager to
trigger the lifespan events that initialize app.state services.
"""
from fastapi.testclient import TestClient
from src.api.app import app
from src.services.security import create_access_token
import uuid


class TestFinanceAPIRoutes:
    def setup_method(self):
        """Create a fresh TestClient for each test.

        Using TestClient as a context manager ensures the lifespan
        events fire, which initializes app.state with service instances.
        """
        self.client = TestClient(app, raise_server_exceptions=True)
        self.client.__enter__()
        unique = uuid.uuid4().hex[:8]
        self.username = f"user_{unique}"
        self.password = "StrongPass123"

        signup_resp = self.client.post(
            "/api/v1/auth/signup",
            json={"username": self.username, "password": self.password},
        )
        assert signup_resp.status_code == 200
        token = signup_resp.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {token}"}

    def teardown_method(self):
        """Clean up the TestClient after each test."""
        self.client.__exit__(None, None, None)

    def test_add_transaction(self):
        """Adding a transaction should return it with an ID and date."""
        resp = self.client.post(
            "/api/v1/transactions",
            json={"amount": 42.50, "category": "food", "description": "Lunch"},
            headers=self.headers,
        )
        assert resp.status_code == 200
        assert resp.json()["amount"] == 42.50

    def test_list_transactions(self):
        """Listing transactions should return a list."""
        resp = self.client.get("/api/v1/transactions", headers=self.headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_summary(self):
        """Summary should include total_spending and by_category."""
        resp = self.client.get("/api/v1/summary", headers=self.headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_spending" in data
        assert "by_category" in data

    def test_chat(self):
        """Chat should return a reply based on transaction data."""
        resp = self.client.post(
            "/api/v1/chat",
            json={"message": "How am I spending?"},
            headers=self.headers,
        )
        assert resp.status_code == 200
        assert "reply" in resp.json()
        assert isinstance(resp.json().get("insights"), list)

    def test_search_memory(self):
        """Memory search should return results (possibly empty)."""
        resp = self.client.get("/api/v1/memory/search?query=tx", headers=self.headers)
        assert resp.status_code == 200

    def test_login(self):
        """Existing user should be able to login and get a token."""
        resp = self.client.post(
            "/api/v1/auth/login",
            json={"username": self.username, "password": self.password},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_unauthorized_access(self):
        """Protected routes should reject requests without a bearer token."""
        resp = self.client.get("/api/v1/transactions")
        assert resp.status_code == 401

    def test_signup_duplicate_username(self):
        """Signing up twice with same username should return 400."""
        resp = self.client.post(
            "/api/v1/auth/signup",
            json={"username": self.username, "password": self.password},
        )
        assert resp.status_code == 400

    def test_login_invalid_password(self):
        """Invalid password should return 401."""
        resp = self.client.post(
            "/api/v1/auth/login",
            json={"username": self.username, "password": "wrongpass"},
        )
        assert resp.status_code == 401

    def test_invalid_token_rejected(self):
        """Malformed token should be rejected by current-user dependency."""
        headers = {"Authorization": "Bearer not.a.valid.token"}
        resp = self.client.get("/api/v1/transactions", headers=headers)
        assert resp.status_code == 401

    def test_unknown_user_token_rejected(self):
        """Validly signed token for missing user should be rejected."""
        token = create_access_token("non-existent-user-id")
        headers = {"Authorization": f"Bearer {token}"}
        resp = self.client.get("/api/v1/transactions", headers=headers)
        assert resp.status_code == 401
