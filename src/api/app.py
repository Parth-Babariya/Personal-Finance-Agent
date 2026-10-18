"""FastAPI application factory for the Personal Finance Agent.

``create_app`` wires services explicitly, so tests and evals can build an app
against an in-memory database and a fake LLM without touching globals.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date

from fastapi import FastAPI

from ..agent.agent import FinanceAgent
from ..agent.llm import LLMClient, OpenAIChatClient
from ..agent.tools import ToolRegistry
from ..config import Settings
from ..services.analytics import AnalyticsService
from ..services.auth_service import AuthService
from ..services.budgets import BudgetService
from ..services.categorize import TransactionCategorizer
from ..services.conversation_store import ConversationStore
from ..services.db import init_db, make_engine, make_session_factory
from ..services.goals import GoalService
from ..services.importer import ImportService
from ..services.insight_generator import SimpleInsightGenerator
from ..services.pending_actions import PendingActionService
from ..services.rate_limit import SlidingWindowLimiter
from ..services.sql_transaction_service import SQLTransactionService
from ..services.usage import UsageService
from .routes import api_router

_UNSET = object()


@dataclass
class Limiters:
    chat: SlidingWindowLimiter
    auth: SlidingWindowLimiter
    login_failures: SlidingWindowLimiter

    @classmethod
    def from_settings(cls, settings: Settings) -> Limiters:
        return cls(
            chat=SlidingWindowLimiter(settings.chat_requests_per_minute, 60),
            auth=SlidingWindowLimiter(settings.auth_requests_per_minute, 60),
            login_failures=SlidingWindowLimiter(settings.login_max_failures, settings.login_lockout_minutes * 60),
        )


def build_llm(settings: Settings) -> LLMClient | None:
    if not settings.llm_api_key:
        return None
    return OpenAIChatClient(settings.llm_api_key, settings.llm_model, base_url=settings.llm_base_url)


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None | object = _UNSET,
    today: Callable[[], date] = date.today,
) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.auto_create_tables:
            init_db(engine)
        yield
        engine.dispose()

    app = FastAPI(
        title="Personal Finance Agent",
        description="LLM tool-calling agent for personal finance, with JWT auth and SQL persistence.",
        version="0.2.0",
        lifespan=lifespan,
    )

    tx_service = SQLTransactionService(session_factory, today=today)
    conversations = ConversationStore(session_factory)
    insight_gen = SimpleInsightGenerator(settings.currency_symbol)
    budgets = BudgetService(session_factory, tx_service, today=today, currency=settings.currency_symbol)
    actions = (
        PendingActionService(session_factory, tx_service, budgets=budgets)
        if settings.require_write_confirmation else None
    )
    analytics = AnalyticsService(tx_service, today=today)
    goals = GoalService(session_factory, tx_service, today=today)
    registry = ToolRegistry.default(
        tx_service, insight_gen, actions, budgets=budgets, analytics=analytics, goals=goals
    )
    llm_client = build_llm(settings) if llm is _UNSET else llm

    app.state.settings = settings
    app.state.tx_service = tx_service
    app.state.auth_service = AuthService(session_factory)
    app.state.conversations = conversations
    app.state.actions = actions
    app.state.budgets = budgets
    app.state.analytics = analytics
    app.state.goals = goals
    app.state.imports = ImportService(
        session_factory, tx_service, TransactionCategorizer(llm_client), today=today  # type: ignore[arg-type]
    )
    app.state.limiters = Limiters.from_settings(settings)
    app.state.agent = FinanceAgent(
        registry,
        conversations,
        llm_client,  # type: ignore[arg-type]
        actions=actions,
        usage=UsageService(session_factory, today=today),
        daily_token_budget=settings.daily_token_budget,
        currency=settings.currency_symbol,
        max_steps=settings.agent_max_steps,
        history_limit=settings.chat_history_limit,
        max_message_chars=settings.max_message_chars,
        timeout_seconds=settings.agent_timeout_seconds,
        today=today,
    )

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, object]:
        return {"status": "ok", "llm_enabled": app.state.agent.llm_enabled}

    return app


def __getattr__(name: str):
    # Lazily build the default app so `uvicorn src.api.app:app` works without
    # import-time side effects for tests that only need `create_app`.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
