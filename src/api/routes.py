"""API routes for the Personal Finance Agent.

This module defines the FastAPI endpoints for managing transactions, chatting
with the finance agent, and searching conversation memory.

Design principles:
  - DIP: All services are injected via FastAPI's dependency injection, making
    them easily swappable for testing or alternative implementations.
  - SRP: Routes only handle HTTP concerns; business logic lives in services.
  - ISP: Memory is split into IMemoryReader/IMemoryWriter interfaces.

Note on dependency injection:
  FastAPI's `Depends()` system creates service instances per-request by default.
  We use `app.state` (set during lifespan) to share instances across requests,
  which gives us explicit lifecycle control. For production, these would be
  backed by a real database; the in-memory versions are for demos and testing.
"""
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from ..services.auth_service import AuthService
from ..services.insight_generator import SimpleInsightGenerator
from ..services.llm_agent import FinanceLLMAgent
from ..services.memory_manager import SessionMemory
from ..services.security import ALGORITHM, SECRET_KEY, create_access_token
from ..services.sql_transaction_service import SQLTransactionService

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# --- Schemas ---
# Pydantic models define the API contract and provide automatic validation.


class TransactionCreate(BaseModel):
    """Schema for creating a new transaction."""
    amount: float = Field(..., description="Transaction amount in dollars")
    category: str = Field(default="uncategorized", description="Spending category")
    description: str = Field(default="", description="Transaction description")


class TransactionResponse(BaseModel):
    """Schema for a transaction returned by the API."""
    id: str
    user_id: str
    amount: float
    category: str
    description: str
    date: str


class SummaryResponse(BaseModel):
    """Spending summary for a time period."""
    period: str
    total_spending: float
    transaction_count: int
    by_category: dict[str, float]


class ChatRequest(BaseModel):
    """Chat message from the user."""
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    """Agent reply with generated insights."""
    reply: str
    insights: list[str] = Field(default_factory=list)


class UserCredentials(BaseModel):
    """Credentials for signup/login endpoints."""

    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=8, max_length=128)


class TokenResponse(BaseModel):
    """JWT access token response model."""

    access_token: str
    token_type: str = "bearer"


# --- Dependency injection helpers ---
# These functions retrieve service instances from FastAPI's app.state,
# which is initialized during the application lifespan. This approach
# gives us explicit control over object lifecycle and makes testing easier
# (override dependencies in tests to inject mocks).

def get_tx_service(request: Request) -> SQLTransactionService:
    """Retrieve the transaction service from app state (DIP)."""
    return request.app.state.tx_service


def get_auth_service(request: Request) -> AuthService:
    """Retrieve the auth service from app state."""
    return request.app.state.auth_service


def get_memory(request: Request) -> SessionMemory:
    """Retrieve the session memory from app state (DIP)."""
    return request.app.state.memory


def get_insight_gen(request: Request) -> SimpleInsightGenerator:
    """Retrieve the insight generator from app state (DIP)."""
    return request.app.state.insight_gen


def get_agent(request: Request) -> FinanceLLMAgent:
    """Retrieve the orchestrated finance agent from app state."""
    return request.app.state.agent


def get_current_user(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> dict[str, str]:
    """Decode JWT token and resolve current user."""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_error
    except JWTError as exc:
        raise credentials_error from exc

    user = auth_service.get_user_by_id(user_id)
    if user is None:
        raise credentials_error
    return {"id": user.id, "username": user.username}


# --- Endpoints ---


@router.post("/auth/signup", response_model=TokenResponse)
async def signup(
    req: UserCredentials,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """Register a new user and return JWT token."""
    try:
        user = auth_service.register_user(req.username, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    req: UserCredentials,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """Authenticate existing user and return JWT token."""
    user = auth_service.authenticate_user(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@router.post("/transactions", response_model=TransactionResponse)
async def add_transaction(
    req: TransactionCreate,
    tx_service: SQLTransactionService = Depends(get_tx_service),
    memory: SessionMemory = Depends(get_memory),
    current_user: dict[str, str] = Depends(get_current_user),
) -> Any:
    """Add a new transaction and store it in session memory.

    The transaction is persisted in the transaction service and also
    cached in session memory for quick recall during chat interactions.
    """
    record = tx_service.add_transaction(req.model_dump(), user_id=current_user["id"])
    # Store in memory so the chat agent can recall recent transactions
    memory.store(f"{current_user['id']}:tx_{record['id']}", record)
    return record


@router.get("/transactions", response_model=list[TransactionResponse])
async def list_transactions(
    limit: int = 50,
    offset: int = 0,
    tx_service: SQLTransactionService = Depends(get_tx_service),
    current_user: dict[str, str] = Depends(get_current_user),
) -> Any:
    """List transactions with pagination.

    Args:
        limit: Maximum number of transactions to return (default: 50).
        offset: Number of transactions to skip (for pagination).
    """
    return tx_service.get_transactions(limit=limit, offset=offset, user_id=current_user["id"])


@router.get("/summary", response_model=SummaryResponse)
async def get_summary(
    period: str = "month",
    tx_service: SQLTransactionService = Depends(get_tx_service),
    current_user: dict[str, str] = Depends(get_current_user),
) -> Any:
    """Get spending summary for a time period."""
    return tx_service.get_summary(period, user_id=current_user["id"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    memory: SessionMemory = Depends(get_memory),
    agent: FinanceLLMAgent = Depends(get_agent),
    current_user: dict[str, str] = Depends(get_current_user),
) -> ChatResponse:
    """Chat with the finance agent.

    Fetches ALL transactions (not just the first page) to ensure the
    agent's insights reflect the complete financial picture. The analysis
    and conversation are stored in memory for future reference.
    """
    result = await agent.respond(req.message, user_id=current_user["id"])
    insights = result.get("insights", [])

    # Store conversation in memory for recall in future interactions
    memory.store(
        f"{current_user['id']}:chat_{memory.size}",
        {"message": req.message, "insights": insights},
    )

    return ChatResponse(reply=result.get("reply", ""), insights=insights)


@router.get("/memory/search")
async def search_memory(
    query: str,
    limit: int = 5,
    memory: SessionMemory = Depends(get_memory),
    current_user: dict[str, str] = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Search through agent memory by keyword.

    Useful for recalling past conversations or finding specific transactions
    that were discussed in earlier chat sessions.
    """
    return memory.search(query, limit=limit, namespace=current_user["id"])
