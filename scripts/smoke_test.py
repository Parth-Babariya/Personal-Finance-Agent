"""End-to-end smoke test against a running API (used by CI against the Docker Compose stack).

    python scripts/smoke_test.py http://localhost:8003

Standard library only, so it runs without installing the project. Exercises every feature that
touches the database (transactions, chat history, pending actions, budgets, goals, analytics,
CSV import) so a Postgres-specific problem surfaces here rather than in production.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
import uuid
from typing import Any

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8003").rstrip("/")
API = f"{BASE}/api/v1"
TOKEN = ""


def call(method: str, path: str, body: Any = None, *, raw: bytes | None = None,
         content_type: str = "application/json", expect: int = 200) -> Any:
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(f"{API}{path}" if path.startswith("/") else path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", content_type)
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status, text = resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        status, text = exc.code, exc.read().decode()
    if status != expect:
        raise SystemExit(f"FAIL {method} {path}: expected {expect}, got {status}: {text[:500]}")
    return json.loads(text) if text else None


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL {message}")
    print(f"ok   {message}")


def multipart(field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def main() -> None:
    global TOKEN
    health = call("GET", f"{BASE}/health")
    check(health["status"] == "ok", f"health ({'LLM' if health['llm_enabled'] else 'keyword fallback'} mode)")

    user = f"smoke_{uuid.uuid4().hex[:8]}"
    TOKEN = call("POST", "/auth/signup", {"username": user, "password": "StrongPass123"}, expect=201)["access_token"]
    check(bool(TOKEN), "signup returns a JWT")

    call("POST", "/transactions", {"amount": 450, "category": "food", "description": "Lunch"}, expect=201)
    call("POST", "/transactions", {"amount": 65000, "type": "income", "category": "salary"}, expect=201)
    summary = call("GET", "/summary?period=this_month")
    check(summary["total_expense"] == 450.0 and summary["net"] == 64550.0, "Numeric amounts and summary")

    call("PUT", "/budgets/food", {"monthly_limit": 500})
    alert = call("POST", "/transactions", {"amount": 10.5, "category": "food"}, expect=201)["budget_alert"]
    check(alert is not None and "92%" in alert, "budget alert on add")
    check(call("GET", "/budgets/status")[0]["state"] == "warning", "budget status")

    goal = call("POST", "/goals", {"name": "Trip", "target_amount": 1000}, expect=201)
    check(call("POST", f"/goals/{goal['id']}/contributions", {"amount": 250})["progress_pct"] == 25.0, "goals")

    check(len(call("GET", "/analytics/trend?months=3")["months"]) == 3, "analytics trend")
    call("GET", "/analytics/recurring")
    call("GET", "/analytics/unusual?period=this_month")

    reply = call("POST", "/chat", {"message": "How much did I spend on food this month?"})
    check("460.50" in reply["reply"], f"chat answers from the database ({reply['mode']})")

    proposal = call("POST", "/chat", {"message": "I spent 300 on uber today"})
    pending = proposal["pending_actions"]
    if pending:  # proposal-based write; the LLM may phrase it differently but must not save yet
        check(call("POST", f"/actions/{pending[0]['id']}/confirm")["status"] == "confirmed", "confirm write")
    check(any(t["amount"] == 300.0 for t in call("GET", "/transactions")), "write saved after confirmation")
    check(len(call("GET", "/chat/history")) >= 4, "chat history persisted")

    csv_body, ctype = multipart("file", "statement.csv",
                                b"Date,Narration,Withdrawal Amt.,Deposit Amt.\n"
                                b"01/01/26,NETFLIX SUBSCRIPTION,649.00,\n02/01/26,SWIGGY ORDER,250.00,\n")
    preview = call("POST", "/imports", raw=csv_body, content_type=ctype, expect=201)
    check([r["category"] for r in preview["rows"]] == ["subscriptions", "food"], "CSV import preview")
    check(call("POST", f"/imports/{preview['id']}/commit", {"rows": []})["imported"] == 2, "CSV import commit")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
