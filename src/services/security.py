"""Security helpers for password hashing and JWT token handling."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))


def hash_password(password: str) -> str:
    """Hash plain text password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Validate plain text password against hash."""
    return pwd_context.verify(plain_password, password_hash)


def create_access_token(subject: str, expires_minutes: int | None = None) -> str:
    """Create a signed JWT access token for the provided subject."""
    minutes = expires_minutes or ACCESS_TOKEN_EXPIRE_MINUTES
    expire = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    payload: dict[str, Any] = {"sub": subject, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
