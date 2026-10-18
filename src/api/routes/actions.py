"""Confirm or cancel writes the agent proposed."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_actions, get_current_user_id
from src.api.schemas import ActionResponse
from src.workflows.approvals import ActionError, PendingActionService

router = APIRouter(prefix="/actions", tags=["actions"])


def _resolve(fn: Callable[[str, str], dict[str, Any] | None], action_id: str, user_id: str) -> Any:
    try:
        action = fn(action_id, user_id)
    except ActionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if action is None:
        raise HTTPException(status_code=404, detail="Action not found")
    return action


@router.get("", response_model=list[ActionResponse])
def list_open_actions(
    actions: PendingActionService = Depends(get_actions),
    user_id: str = Depends(get_current_user_id),
) -> Any:
    return actions.list_open(user_id)


@router.post("/{action_id}/confirm", response_model=ActionResponse)
def confirm_action(
    action_id: str,
    actions: PendingActionService = Depends(get_actions),
    user_id: str = Depends(get_current_user_id),
) -> Any:
    return _resolve(actions.confirm, action_id, user_id)


@router.post("/{action_id}/cancel", response_model=ActionResponse)
def cancel_action(
    action_id: str,
    actions: PendingActionService = Depends(get_actions),
    user_id: str = Depends(get_current_user_id),
) -> Any:
    return _resolve(actions.cancel, action_id, user_id)
