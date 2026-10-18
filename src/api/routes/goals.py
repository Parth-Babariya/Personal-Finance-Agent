"""Savings goals."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status

from src.api.deps import get_current_user_id, get_goals
from src.api.schemas import GoalContribution, GoalCreate
from src.services.goals import GoalService

router = APIRouter(prefix="/goals", tags=["goals"])


@router.get("")
def list_goals(
    goals: GoalService = Depends(get_goals),
    user_id: str = Depends(get_current_user_id),
) -> list[dict[str, Any]]:
    return goals.list(user_id)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_goal(
    req: GoalCreate,
    goals: GoalService = Depends(get_goals),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    try:
        return goals.create(user_id, req.name, req.target_amount, req.target_date, req.saved_amount)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{goal_id}/contributions")
def contribute(
    goal_id: str,
    req: GoalContribution,
    goals: GoalService = Depends(get_goals),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    try:
        goal = goals.contribute(goal_id, user_id, req.amount)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal


@router.delete("/{goal_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_goal(
    goal_id: str,
    goals: GoalService = Depends(get_goals),
    user_id: str = Depends(get_current_user_id),
) -> Response:
    if not goals.delete(goal_id, user_id):
        raise HTTPException(status_code=404, detail="Goal not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
