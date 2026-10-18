"""Application settings loaded from environment variables."""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. Build with ``Settings.from_env()`` or directly in tests."""

    database_url: str = "sqlite:///./finance_agent.db"
    jwt_secret_key: str = ""
    jwt_expire_minutes: int = 60
    app_env: str = "development"
    auto_create_tables: bool = True

    # LLM (any OpenAI-compatible endpoint: OpenAI, Groq, Gemini, Ollama, ...)
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"

    # Agent behaviour
    currency_symbol: str = "₹"
    agent_max_steps: int = 5
    chat_history_limit: int = 10
    max_message_chars: int = 2000
    agent_timeout_seconds: float = 45.0
    require_write_confirmation: bool = True

    # Abuse and cost controls (0 disables a limit)
    chat_requests_per_minute: int = 20
    auth_requests_per_minute: int = 10
    login_max_failures: int = 5
    login_lockout_minutes: int = 15
    daily_token_budget: int = 200_000

    @classmethod
    def from_env(cls) -> Settings:
        app_env = os.getenv("APP_ENV", "development").lower()
        secret = os.getenv("JWT_SECRET_KEY", "")
        if not secret:
            if app_env == "production":
                raise RuntimeError("JWT_SECRET_KEY must be set when APP_ENV=production")
            # Ephemeral secret: safe default for local dev, tokens reset on restart.
            secret = secrets.token_urlsafe(48)
            logger.warning("JWT_SECRET_KEY not set; using an ephemeral secret (dev only).")

        return cls(
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            jwt_secret_key=secret,
            jwt_expire_minutes=int(os.getenv("JWT_EXPIRE_MINUTES", "60")),
            app_env=app_env,
            auto_create_tables=_env_bool("AUTO_CREATE_TABLES", True),
            llm_api_key=os.getenv("OPENAI_API_KEY") or None,
            llm_base_url=os.getenv("OPENAI_BASE_URL") or None,
            llm_model=os.getenv("OPENAI_MODEL", cls.llm_model),
            currency_symbol=os.getenv("CURRENCY_SYMBOL", cls.currency_symbol),
            agent_max_steps=int(os.getenv("AGENT_MAX_STEPS", "5")),
            chat_history_limit=int(os.getenv("CHAT_HISTORY_LIMIT", "10")),
            agent_timeout_seconds=float(os.getenv("AGENT_TIMEOUT_SECONDS", "45")),
            require_write_confirmation=_env_bool("REQUIRE_WRITE_CONFIRMATION", True),
            chat_requests_per_minute=int(os.getenv("CHAT_REQUESTS_PER_MINUTE", "20")),
            auth_requests_per_minute=int(os.getenv("AUTH_REQUESTS_PER_MINUTE", "10")),
            login_max_failures=int(os.getenv("LOGIN_MAX_FAILURES", "5")),
            login_lockout_minutes=int(os.getenv("LOGIN_LOCKOUT_MINUTES", "15")),
            daily_token_budget=int(os.getenv("DAILY_TOKEN_BUDGET", "200000")),
        )
