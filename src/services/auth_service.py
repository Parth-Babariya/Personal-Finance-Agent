"""Authentication service backed by SQLAlchemy user storage."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from .models import User
from .security import hash_password, verify_password


class AuthService:
    """User creation and credential verification service."""

    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    def get_user_by_id(self, user_id: str) -> User | None:
        """Fetch a user by unique id."""
        with self._session_factory() as session:
            stmt = select(User).where(User.id == user_id)
            return session.scalar(stmt)

    def get_user_by_username(self, username: str) -> User | None:
        """Fetch a user by username."""
        with self._session_factory() as session:
            stmt = select(User).where(User.username == username)
            return session.scalar(stmt)

    def register_user(self, username: str, password: str) -> User:
        """Create a new user if username is not already taken."""
        with self._session_factory() as session:
            existing = session.scalar(select(User).where(User.username == username))
            if existing is not None:
                raise ValueError("Username already exists")

            user = User(username=username, password_hash=hash_password(password))
            session.add(user)
            session.commit()
            session.refresh(user)
            return user

    def authenticate_user(self, username: str, password: str) -> User | None:
        """Validate username/password and return user on success."""
        with self._session_factory() as session:
            user = session.scalar(select(User).where(User.username == username))
            if user is None:
                return None
            if not verify_password(password, user.password_hash):
                return None
            return user
