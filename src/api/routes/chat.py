"""Chat endpoints: blocking JSON, Server-Sent Events streaming, and history."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from ...agent.agent import FinanceAgent
from ...services.conversation_store import ConversationStore
from ..deps import get_agent, get_conversations, get_current_user_id, rate_limit_chat
from ..schemas import ChatMessageResponse, ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    agent: FinanceAgent = Depends(get_agent),
    user_id: str = Depends(rate_limit_chat),
) -> Any:
    result = await agent.respond(req.message, user_id=user_id)
    return {
        "reply": result.reply,
        "mode": result.mode,
        "tool_calls": result.tool_calls,
        "steps": result.steps,
        "latency_ms": result.latency_ms,
        "fallback_reason": result.fallback_reason,
        "redacted": result.redacted,
        "pending_actions": result.pending_actions,
    }


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    agent: FinanceAgent = Depends(get_agent),
    user_id: str = Depends(rate_limit_chat),
) -> StreamingResponse:
    """Stream agent events as SSE: ``token``, ``tool_call``, ``tool_result``, ``done``."""

    async def events() -> AsyncIterator[str]:
        try:
            async for event in agent.stream(req.message, user_id=user_id):
                yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
        except Exception:  # surface failures to the client instead of a broken stream
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'detail': 'Agent failed'})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chat/history", response_model=list[ChatMessageResponse])
def chat_history(
    limit: int = Query(50, ge=1, le=200),
    conversations: ConversationStore = Depends(get_conversations),
    user_id: str = Depends(get_current_user_id),
) -> Any:
    return conversations.recent(user_id, limit)


@router.delete("/chat/history")
def clear_chat_history(
    conversations: ConversationStore = Depends(get_conversations),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, int]:
    return {"deleted": conversations.clear(user_id)}


@router.get("/memory/search", response_model=list[ChatMessageResponse])
def search_memory(
    query: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(5, ge=1, le=50),
    conversations: ConversationStore = Depends(get_conversations),
    user_id: str = Depends(get_current_user_id),
) -> Any:
    """Keyword search over the user's past conversation."""
    return conversations.search(user_id, query, limit)
