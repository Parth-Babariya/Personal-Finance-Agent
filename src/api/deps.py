"""FastAPI dependencies: services from ``app.state``, the current user, rate limits."""

from __future__ import annotations

import math

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from src.agents.finance_agent import FinanceAgent
from src.core.config import Settings
from src.core.security import decode_access_token
from src.memory.conversation_store import ConversationStore
from src.services.analytics import AnalyticsService
from src.services.auth import AuthService
from src.services.budgets import BudgetService
from src.services.goals import GoalService
from src.services.transactions import SQLTransactionService
from src.workflows.approvals import PendingActionService
from src.workflows.statement_import import ImportService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_tx_service(request: Request) -> SQLTransactionService:
    return request.app.state.tx_service


def get_auth_service(request: Request) -> AuthService:
    return request.app.state.auth_service


def get_conversations(request: Request) -> ConversationStore:
    return request.app.state.conversations


def get_actions(request: Request) -> PendingActionService:
    if request.app.state.actions is None:
        raise HTTPException(status_code=404, detail="Write confirmation is disabled")
    return request.app.state.actions


def get_analytics(request: Request) -> AnalyticsService:
    return request.app.state.analytics


def get_imports(request: Request) -> ImportService:
    return request.app.state.imports


def get_goals(request: Request) -> GoalService:
    return request.app.state.goals


def get_budgets(request: Request) -> BudgetService:
    return request.app.state.budgets


def get_agent(request: Request) -> FinanceAgent:
    return request.app.state.agent


def get_current_user_id(
    token: str = Depends(oauth2_scheme),
    settings: Settings = Depends(get_settings),
    auth_service: AuthService = Depends(get_auth_service),
) -> str:
    user_id = decode_access_token(token, settings.jwt_secret_key)
    if user_id is None or auth_service.get_user_by_id(user_id) is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


def too_many_requests(retry_after: float, detail: str = "Too many requests") -> None:
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=detail,
        headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
    )


def rate_limit_auth(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    wait = request.app.state.limiters.auth.hit(client)
    if wait is not None:
        too_many_requests(wait)


def rate_limit_chat(request: Request, user_id: str = Depends(get_current_user_id)) -> str:
    """Resolve the current user and enforce the per-user chat rate limit."""
    wait = request.app.state.limiters.chat.hit(user_id)
    if wait is not None:
        too_many_requests(wait, "Chat rate limit reached. Please wait a moment.")
    return user_id
