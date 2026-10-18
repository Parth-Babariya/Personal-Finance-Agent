"""Pydantic request/response models: the public API contract."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..agent.tools import Category, TxType


class UserCredentials(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(..., min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TransactionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(..., gt=0, le=10_000_000, description="Positive amount")
    type: TxType = "expense"
    category: Category = "other"
    description: str = Field(default="", max_length=500)
    date: dt.date | None = Field(default=None, description="Defaults to today")


class TransactionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float | None = Field(default=None, gt=0, le=10_000_000)
    type: TxType | None = None
    category: Category | None = None
    description: str | None = Field(default=None, max_length=500)
    date: dt.date | None = None


class TransactionResponse(BaseModel):
    id: str
    user_id: str
    type: TxType
    amount: float
    category: str
    description: str
    date: dt.date


class SummaryResponse(BaseModel):
    period: str
    start_date: dt.date | None
    end_date: dt.date | None
    category: str | None = None
    total_expense: float
    total_income: float
    net: float
    transaction_count: int
    by_category: dict[str, float]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ToolCallInfo(BaseModel):
    name: str
    arguments: dict[str, Any]
    ok: bool


class PendingActionInfo(BaseModel):
    id: str
    kind: str
    payload: dict[str, Any]
    possible_duplicates: list[dict[str, Any]] = []


class ChatResponse(BaseModel):
    reply: str
    mode: Literal["llm", "fallback", "action"]
    tool_calls: list[ToolCallInfo]
    steps: int
    latency_ms: float
    fallback_reason: str | None = None
    redacted: list[str] = []
    pending_actions: list[PendingActionInfo] = []


class ActionResponse(BaseModel):
    id: str
    kind: str
    payload: dict[str, Any]
    status: str
    created_at: str
    expires_at: str
    result: dict[str, Any] | None = None


class ChatMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    meta: dict[str, Any]
    created_at: str
