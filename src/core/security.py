"""Security helpers for password hashing and JWT token handling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def create_access_token(subject: str, secret_key: str, expires_minutes: int = 60) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=expires_minutes)
    return jwt.encode({"sub": subject, "exp": expire}, secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str, secret_key: str) -> str | None:
    """Return the token subject, or ``None`` if the token is invalid or expired."""
    try:
        payload = jwt.decode(token, secret_key, algorithms=[ALGORITHM])
    except JWTError:
        return None
    subject = payload.get("sub")
    return subject if isinstance(subject, str) else None
