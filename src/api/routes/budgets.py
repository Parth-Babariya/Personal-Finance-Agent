"""Monthly category budgets."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Response, status

from ...agent.tools import ExpenseCategory
from ...services.budgets import BudgetService
from ...services.periods import resolve_period
from ..deps import get_budgets, get_current_user_id
from ..schemas import BudgetSet, BudgetStatus

router = APIRouter(prefix="/budgets", tags=["budgets"])


@router.get("")
def list_budgets(
    budgets: BudgetService = Depends(get_budgets),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, float]:
    return budgets.list_budgets(user_id)


@router.get("/status", response_model=list[BudgetStatus])
def budget_status(
    month: Literal["this_month", "last_month"] = "this_month",
    budgets: BudgetService = Depends(get_budgets),
    user_id: str = Depends(get_current_user_id),
) -> Any:
    start, _ = resolve_period(month, budgets.today())
    return budgets.status(user_id, month=start)


@router.put("/{category}")
def set_budget(
    category: ExpenseCategory,
    req: BudgetSet,
    budgets: BudgetService = Depends(get_budgets),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    return budgets.set_budget(user_id, category, req.monthly_limit)


@router.delete("/{category}", status_code=status.HTTP_204_NO_CONTENT)
def delete_budget(
    category: ExpenseCategory,
    budgets: BudgetService = Depends(get_budgets),
    user_id: str = Depends(get_current_user_id),
) -> Response:
    if not budgets.delete_budget(user_id, category):
        raise HTTPException(status_code=404, detail="No budget for this category")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
