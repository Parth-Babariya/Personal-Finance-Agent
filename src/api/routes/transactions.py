"""Transaction CRUD and summary endpoints.

Handlers are plain ``def`` so FastAPI runs the synchronous SQLAlchemy calls in
its threadpool instead of blocking the event loop.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from ...agent.tools import Category, TxType
from ...services.budgets import BudgetService
from ...services.periods import describe_range, normalize_period, resolve_period
from ...services.sql_transaction_service import SQLTransactionService
from ..deps import get_budgets, get_current_user_id, get_tx_service
from ..schemas import SummaryResponse, TransactionCreate, TransactionResponse, TransactionUpdate

router = APIRouter(tags=["transactions"])


def _date_range(
    tx_service: SQLTransactionService,
    period: str | None,
    start_date: date | None,
    end_date: date | None,
) -> tuple[date | None, date | None, str]:
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be on or before end_date")
    if start_date or end_date:
        return start_date, end_date, "custom"
    if period is None:
        return None, None, "all_time"
    try:
        key = normalize_period(period)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    start, end = resolve_period(key, tx_service.today())
    return start, end, key


@router.post("/transactions", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED)
def add_transaction(
    req: TransactionCreate,
    tx_service: SQLTransactionService = Depends(get_tx_service),
    budgets: BudgetService = Depends(get_budgets),
    user_id: str = Depends(get_current_user_id),
) -> Any:
    record = tx_service.add_transaction(req.model_dump(), user_id=user_id)
    if record["type"] == "expense":
        record["budget_alert"] = budgets.alert_for(user_id, record["category"], date.fromisoformat(record["date"]))
    return record


@router.get("/transactions", response_model=list[TransactionResponse])
def list_transactions(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    period: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    category: Category | None = None,
    type: TxType | None = None,
    q: str | None = Query(None, max_length=100, description="Search descriptions"),
    tx_service: SQLTransactionService = Depends(get_tx_service),
    user_id: str = Depends(get_current_user_id),
) -> Any:
    start, end, _ = _date_range(tx_service, period, start_date, end_date)
    return tx_service.list_transactions(
        user_id, limit=limit, offset=offset, start=start, end=end, category=category, type=type, query=q
    )


@router.get("/transactions/{transaction_id}", response_model=TransactionResponse)
def get_transaction(
    transaction_id: str,
    tx_service: SQLTransactionService = Depends(get_tx_service),
    user_id: str = Depends(get_current_user_id),
) -> Any:
    record = tx_service.get_transaction(transaction_id, user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return record


@router.patch("/transactions/{transaction_id}", response_model=TransactionResponse)
def update_transaction(
    transaction_id: str,
    req: TransactionUpdate,
    tx_service: SQLTransactionService = Depends(get_tx_service),
    user_id: str = Depends(get_current_user_id),
) -> Any:
    record = tx_service.update_transaction(transaction_id, req.model_dump(exclude_unset=True), user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return record


@router.delete("/transactions/{transaction_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_transaction(
    transaction_id: str,
    tx_service: SQLTransactionService = Depends(get_tx_service),
    user_id: str = Depends(get_current_user_id),
) -> Response:
    if not tx_service.delete_transaction(transaction_id, user_id):
        raise HTTPException(status_code=404, detail="Transaction not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/summary", response_model=SummaryResponse)
def get_summary(
    period: str = "this_month",
    start_date: date | None = None,
    end_date: date | None = None,
    category: Category | None = None,
    tx_service: SQLTransactionService = Depends(get_tx_service),
    user_id: str = Depends(get_current_user_id),
) -> Any:
    start, end, label = _date_range(tx_service, period, start_date, end_date)
    summary = tx_service.get_summary(user_id, start=start, end=end, category=category)
    return {"period": label if label != "custom" else describe_range(start, end), **summary}
