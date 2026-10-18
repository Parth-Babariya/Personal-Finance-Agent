"""Spending trends, unusual expenses and recurring payments."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_analytics, get_current_user_id
from src.core.types import Category
from src.services.analytics import AnalyticsService
from src.services.periods import describe_range, normalize_period, resolve_period

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/trend")
def monthly_trend(
    months: int = Query(6, ge=2, le=12),
    category: Category | None = None,
    analytics: AnalyticsService = Depends(get_analytics),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    return analytics.monthly_trend(user_id, months, category)


@router.get("/unusual")
def unusual_expenses(
    period: str = "last_30_days",
    analytics: AnalyticsService = Depends(get_analytics),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    try:
        start, end = resolve_period(normalize_period(period), analytics.today())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"period": describe_range(start, end), "unusual": analytics.unusual_expenses(user_id, start, end)}


@router.get("/recurring")
def recurring_payments(
    analytics: AnalyticsService = Depends(get_analytics),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    return analytics.recurring_payments(user_id)
