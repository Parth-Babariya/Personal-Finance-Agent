"""Import bank statements from CSV: parse, categorise, flag duplicates, preview, commit.

Nothing is written until the user reviews the preview and commits it.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from src.core.types import CATEGORIES
from src.db.models import ImportBatch
from src.interfaces.transaction_service import ITransactionService
from src.workflows.categorization import TransactionCategorizer

MAX_BYTES = 1_000_000
MAX_ROWS = 2000
MAX_DESCRIPTION = 200
HEADER_SEARCH_ROWS = 20

_ALIASES = {
    "date": ("date", "txndate", "transactiondate", "trandate", "postingdate", "valuedate", "valuedt"),
    "description": ("description", "narration", "details", "particulars", "remarks", "transactiondetails",
                    "transactionremarks", "merchant", "payee"),
    "amount": ("amount", "amt", "transactionamount", "amountinr", "amountrs", "txnamount"),
    "debit": ("debit", "withdrawal", "withdrawalamt", "withdrawalamount", "debitamount", "dr", "withdrawals"),
    "credit": ("credit", "deposit", "depositamt", "depositamount", "creditamount", "cr", "deposits"),
    "type": ("type", "drcr", "crdr", "transactiontype", "debitcredit"),
}
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y",
                 "%d-%b-%Y", "%d %b %Y", "%d-%b-%y", "%d %b %y", "%b %d, %Y", "%d %B %Y")


class StatementError(ValueError):
    """Raised for files that cannot be imported at all."""


def _norm(header: str) -> str:
    return re.sub(r"[^a-z]", "", header.lower())


def _clean_text(value: str) -> str:
    value = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    return re.sub(r"\s+", " ", value).strip()[:MAX_DESCRIPTION]


def parse_amount(raw: str) -> float | None:
    text = raw.strip()
    if not text or text in {"-", "--"}:
        return None
    negative = text.startswith("(") and text.endswith(")") or text.startswith("-")
    suffix = re.search(r"\b(dr|cr)\.?$", text, re.IGNORECASE)
    digits = re.sub(r"[^\d.]", "", text)
    if not digits or digits.count(".") > 1:
        return None
    value = float(digits)
    if negative or (suffix and suffix.group(1).lower() == "dr"):
        value = -value
    return value


def parse_date(raw: str) -> date | None:
    text = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _find_columns(header: list[str]) -> dict[str, int]:
    normalized = [_norm(h) for h in header]
    columns: dict[str, int] = {}
    for field, aliases in _ALIASES.items():
        for alias in aliases:  # aliases are in preference order
            if alias in normalized:
                columns[field] = normalized.index(alias)
                break
    return columns


def parse_statement(content: bytes, today: date) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (rows, errors). Each row has date, description, amount (positive) and type."""
    if len(content) > MAX_BYTES:
        raise StatementError(f"File is larger than {MAX_BYTES // 1000} KB")
    text = content.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    table = list(csv.reader(io.StringIO(text), dialect))

    header_index, columns = None, {}
    for i, row in enumerate(table[:HEADER_SEARCH_ROWS]):  # banks often add a preamble above the header
        found = _find_columns(row)
        if "date" in found and ("amount" in found or "debit" in found or "credit" in found):
            header_index, columns = i, found
            break
    if header_index is None:
        raise StatementError("Could not find a header with a date column and an amount or debit/credit columns")

    body = [r for r in table[header_index + 1:] if any(cell.strip() for cell in r)]
    if len(body) > MAX_ROWS:
        raise StatementError(f"Too many rows (max {MAX_ROWS})")

    signed_file = "amount" in columns and "type" not in columns and any(
        (parse_amount(r[columns["amount"]]) or 0) < 0 for r in body if len(r) > columns["amount"]
    )

    def cell(row: list[str], field: str) -> str:
        index = columns.get(field)
        return row[index] if index is not None and index < len(row) else ""

    rows, errors = [], []
    for line_no, raw in enumerate(body, start=header_index + 2):
        when = parse_date(cell(raw, "date"))
        if when is None:
            errors.append({"line": line_no, "error": f"Unrecognised date '{cell(raw, 'date')}'"})
            continue
        if when > today:
            errors.append({"line": line_no, "error": "Date is in the future"})
            continue

        debit, credit = parse_amount(cell(raw, "debit")), parse_amount(cell(raw, "credit"))
        if debit or credit:
            amount, type_ = (abs(debit), "expense") if debit else (abs(credit), "income")
        else:
            value = parse_amount(cell(raw, "amount"))
            if value is None:
                errors.append({"line": line_no, "error": "Missing amount"})
                continue
            marker = _norm(cell(raw, "type"))
            if marker:
                type_ = "income" if marker.startswith(("cr", "credit", "deposit")) else "expense"
            elif signed_file:
                type_ = "expense" if value < 0 else "income"
            else:
                type_ = "expense"
            amount = abs(value)
        if amount == 0:
            errors.append({"line": line_no, "error": "Zero amount"})
            continue
        rows.append({"date": when.isoformat(), "description": _clean_text(cell(raw, "description")),
                     "amount": round(amount, 2), "type": type_})
    return rows, errors


class ImportService:
    def __init__(
        self,
        session_factory: sessionmaker,
        tx_service: ITransactionService,
        categorizer: TransactionCategorizer,
        today: Callable[[], date] = date.today,
    ):
        self._session_factory = session_factory
        self._tx = tx_service
        self._categorizer = categorizer
        self._today = today

    async def preview(self, user_id: str, filename: str, content: bytes) -> dict[str, Any]:
        rows, errors = parse_statement(content, self._today())
        if not rows:
            raise StatementError("No valid transactions found in the file")
        labels = await self._categorizer.categorize(rows)
        existing = self._existing_keys(user_id, rows)
        for index, (row, (category, source)) in enumerate(zip(rows, labels, strict=True)):
            duplicate = (row["date"], row["amount"], row["type"]) in existing
            row.update(index=index, category=category, category_source=source, duplicate=duplicate,
                       include=not duplicate)

        with self._session_factory() as session:
            batch = ImportBatch(user_id=user_id, filename=filename[:120], rows=rows, errors=errors[:100])
            session.add(batch)
            session.commit()
            return self._to_dict(batch)

    def get(self, user_id: str, import_id: str) -> dict[str, Any] | None:
        with self._session_factory() as session:
            batch = self._batch(session, user_id, import_id)
            return self._to_dict(batch) if batch else None

    def commit(self, user_id: str, import_id: str, overrides: list[dict[str, Any]]) -> dict[str, Any] | None:
        with self._session_factory() as session:
            batch = self._batch(session, user_id, import_id)
            if batch is None:
                return None
            if batch.status != "pending":
                raise ValueError(f"Import is already {batch.status}")
            rows = [dict(r) for r in batch.rows]
            for change in overrides:
                index = change["index"]
                if not 0 <= index < len(rows):
                    raise ValueError(f"Row index {index} out of range")
                if change.get("category") is not None:
                    if change["category"] not in CATEGORIES:
                        raise ValueError(f"Unknown category '{change['category']}'")
                    rows[index]["category"] = change["category"]
                if change.get("include") is not None:
                    rows[index]["include"] = bool(change["include"])

            imported = 0
            for row in rows:
                if row["include"]:
                    self._tx.add_transaction(
                        {"amount": row["amount"], "type": row["type"], "category": row["category"],
                         "description": row["description"], "date": date.fromisoformat(row["date"])},
                        user_id,
                    )
                    imported += 1
            batch.rows, batch.status, batch.imported_count = rows, "committed", imported
            session.commit()
            return {"id": batch.id, "status": batch.status, "imported": imported, "skipped": len(rows) - imported}

    def discard(self, user_id: str, import_id: str) -> bool:
        with self._session_factory() as session:
            batch = self._batch(session, user_id, import_id)
            if batch is None or batch.status != "pending":
                return False
            batch.status = "discarded"
            session.commit()
            return True

    def _existing_keys(self, user_id: str, rows: list[dict[str, Any]]) -> set[tuple[str, float, str]]:
        dates = [date.fromisoformat(r["date"]) for r in rows]
        existing = self._tx.list_transactions(user_id, limit=0, start=min(dates), end=max(dates))
        return {(t["date"], t["amount"], t["type"]) for t in existing}

    @staticmethod
    def _batch(session, user_id: str, import_id: str) -> ImportBatch | None:
        return session.scalar(select(ImportBatch).where(ImportBatch.id == import_id, ImportBatch.user_id == user_id))

    @staticmethod
    def _to_dict(batch: ImportBatch) -> dict[str, Any]:
        rows = batch.rows
        included = [r for r in rows if r["include"]]
        return {
            "id": batch.id,
            "filename": batch.filename,
            "status": batch.status,
            "rows": rows,
            "errors": batch.errors or [],
            "summary": {
                "rows": len(rows),
                "included": len(included),
                "duplicates": sum(r["duplicate"] for r in rows),
                "expense_total": round(sum(r["amount"] for r in included if r["type"] == "expense"), 2),
                "income_total": round(sum(r["amount"] for r in included if r["type"] == "income"), 2),
                "categorized_by": {
                    source: sum(r["category_source"] == source for r in rows) for source in ("rule", "llm", "default")
                },
            },
        }
