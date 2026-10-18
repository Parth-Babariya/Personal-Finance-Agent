"""FastAPI dependencies: services from ``app.state`` and the current user."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from ..agent.agent import FinanceAgent
from ..config import Settings
from ..services.auth_service import AuthService
from ..services.conversation_store import ConversationStore
from ..services.pending_actions import PendingActionService
from ..services.security import decode_access_token
from ..services.sql_transaction_service import SQLTransactionService

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
