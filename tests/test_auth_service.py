"""Unit tests for AuthService using an isolated in-memory SQLite database."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.services.auth_service import AuthService
from src.services.db import Base


class TestAuthService:
    """Tests registration, lookup, and authentication branches."""

    def setup_method(self):
        engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            future=True,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(bind=engine)
        self.session_factory = sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
            future=True,
        )
        self.auth = AuthService(self.session_factory)

    def test_register_and_get_user(self):
        """Registered user should be retrievable by id and username."""
        user = self.auth.register_user("alice", "StrongPass123")

        found_by_id = self.auth.get_user_by_id(user.id)
        found_by_username = self.auth.get_user_by_username("alice")

        assert found_by_id is not None
        assert found_by_id.username == "alice"
        assert found_by_username is not None
        assert found_by_username.id == user.id

    def test_register_duplicate_username_raises(self):
        """Re-registering the same username should fail."""
        self.auth.register_user("bob", "StrongPass123")

        try:
            self.auth.register_user("bob", "AnotherPass456")
            assert False, "Expected ValueError for duplicate username"
        except ValueError as exc:
            assert "already exists" in str(exc)

    def test_authenticate_user_success_and_failure_paths(self):
        """Auth should accept valid credentials and reject invalid ones."""
        self.auth.register_user("charlie", "StrongPass123")

        ok = self.auth.authenticate_user("charlie", "StrongPass123")
        wrong_password = self.auth.authenticate_user("charlie", "wrong-password")
        missing_user = self.auth.authenticate_user("nobody", "StrongPass123")

        assert ok is not None
        assert ok.username == "charlie"
        assert wrong_password is None
        assert missing_user is None
