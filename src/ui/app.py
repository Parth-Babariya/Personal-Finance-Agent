"""Streamlit front-end for the Personal Finance Agent.

A thin client: all finance logic lives in the FastAPI backend. Run with
``streamlit run src/ui/app.py`` after starting the API.
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8003/api/v1")
CURRENCY = os.getenv("CURRENCY_SYMBOL", "₹")
EXPENSE_CATEGORIES = ["food", "groceries", "transport", "shopping", "entertainment", "utilities",
                      "rent", "health", "education", "travel", "subscriptions", "other"]
INCOME_CATEGORIES = ["salary", "freelance", "investment", "gift", "refund"]
PERIODS = {
    "This month": "this_month",
    "Last month": "last_month",
    "Last 30 days": "last_30_days",
    "This year": "this_year",
    "All time": "all_time",
}
SUGGESTIONS = [
    "How much did I spend this month?",
    "Compare this month with last month",
    "What were my biggest expenses this month?",
]

st.set_page_config(page_title="Personal Finance Agent", page_icon=":material/savings:", layout="wide")

if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.messages = []
    st.session_state.pending = []


# --- API helpers ---


def api(method: str, path: str, **kwargs: Any) -> httpx.Response:
    headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
    return httpx.request(method, f"{API_BASE}{path}", headers=headers, timeout=60.0, **kwargs)


def error_detail(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        return resp.text
    if isinstance(detail, list):
        return "; ".join(d.get("msg", str(d)) for d in detail)
    return str(detail)


def money(value: float) -> str:
    return f"{CURRENCY}{value:,.2f}"


def authenticate(action: str, username: str, password: str) -> None:
    try:
        resp = api("POST", f"/auth/{action}", json={"username": username, "password": password})
    except httpx.HTTPError:
        st.error("Cannot reach the API. Is the backend running?")
        return
    if resp.is_success:
        st.session_state.token = resp.json()["access_token"]
        st.session_state.username = username
        history = api("GET", "/chat/history")
        st.session_state.messages = history.json() if history.is_success else []
        pending = api("GET", "/actions")
        st.session_state.pending = pending.json() if pending.is_success else []
        st.rerun()
    else:
        st.error(error_detail(resp))


FALLBACK_NOTES = {
    "llm_error": "AI model unavailable, answered with the basic planner",
    "timeout": "AI model timed out, answered with the basic planner",
    "budget": "Daily AI usage limit reached, answered with the basic planner",
}


def stream_reply(message: str, steps: Any, placeholder: Any) -> tuple[str, dict[str, Any] | None]:
    """Render SSE events live: tool calls as timeline steps, tokens into ``placeholder``.

    Uses a placeholder instead of st.write_stream because the server may send a
    ``reset`` event (grounding retry) that replaces text already shown.
    """
    text, done = "", None
    with httpx.stream(
        "POST", f"{API_BASE}/chat/stream", json={"message": message},
        headers={"Authorization": f"Bearer {st.session_state.token}"}, timeout=120.0,
    ) as resp:
        if not resp.is_success:
            resp.read()
            text = f"Error: {error_detail(resp)}"
            placeholder.warning(text)
            return text, None
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "token":
                text += event["text"]
                placeholder.markdown(text + "▌")
            elif event["type"] == "reset":
                text = ""
                placeholder.empty()
            elif event["type"] == "tool_call":
                with steps:
                    with st.status(f"`{event['name']}`", type="step", state="complete"):
                        st.json(event["arguments"], expanded=False)
            elif event["type"] == "done":
                done = event
            elif event["type"] == "error":
                text += f"\n\nError: {event['detail']}"
    placeholder.markdown(text)
    return text, done


def resolve_action(action: dict[str, Any], verb: str) -> None:
    resp = api("POST", f"/actions/{action['id']}/{verb}")
    t = action["payload"]
    what = f"{t['type']} of {money(t['amount'])} in '{t['category']}' on {t['date']}"
    if resp.is_success:
        content = f"Saved the {what}." if verb == "confirm" else f"Okay, discarded the {what}."
        st.session_state.messages.append({"role": "assistant", "content": content, "meta": {}})
    else:
        st.toast(error_detail(resp), icon=":material/error:")
    st.session_state.pending = [a for a in st.session_state.pending if a["id"] != action["id"]]


# --- login gate ---

if not st.session_state.token:
    st.title("Personal finance agent")
    with st.container(border=True, width=420):
        with st.form("auth"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            with st.container(horizontal=True):
                login = st.form_submit_button("Log in", type="primary")
                signup = st.form_submit_button("Sign up")
        if login or signup:
            authenticate("login" if login else "signup", username, password)
    st.stop()


# --- sidebar ---

with st.sidebar:
    st.caption(f"Signed in as **{st.session_state.username}**")
    view = st.radio(
        "View", ["Chat", "Dashboard", "Transactions", "Budgets & goals", "Import"], label_visibility="collapsed"
    )
    if st.button("Clear chat history", icon=":material/delete_sweep:", width="stretch"):
        api("DELETE", "/chat/history")
        st.session_state.messages = []
        st.rerun()
    if st.button("Log out", icon=":material/logout:", width="stretch"):
        st.session_state.token = None
        st.session_state.messages = []
        st.rerun()


# --- chat ---

if view == "Chat":
    st.title("Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            tools = (msg.get("meta") or {}).get("tools")
            if tools:
                st.caption("Tools: " + ", ".join(f"`{t}`" for t in tools))

    prompt = st.chat_input("Ask about your finances, or log an expense", submit_mode="disable")
    if not st.session_state.messages and not prompt:
        picked = st.pills("Try asking", SUGGESTIONS, label_visibility="collapsed")
        prompt = picked or prompt

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            steps = st.container()
            reply, done = stream_reply(prompt, steps, st.empty())
            if done:
                notes = [f"{done['mode']} · {done['latency_ms']:.0f} ms"]
                if done.get("fallback_reason") in FALLBACK_NOTES:
                    notes.append(FALLBACK_NOTES[done["fallback_reason"]])
                if done.get("redacted"):
                    notes.append("masked: " + ", ".join(done["redacted"]))
                st.caption(" · ".join(notes))
        st.session_state.messages += [
            {"role": "user", "content": (done or {}).get("user_message", prompt)},
            {"role": "assistant", "content": reply,
             "meta": {"tools": [t["name"] for t in (done or {}).get("tool_calls", [])]}},
        ]
        st.session_state.pending = (done or {}).get("pending_actions") or (
            [] if (done or {}).get("mode") == "action" else st.session_state.pending
        )

    for action in st.session_state.pending:
        t = action["payload"]
        with st.container(border=True):
            st.markdown(f"**Record {t['type']}?** {money(t['amount'])} · {t['category']} · {t['date']}"
                        + (f" · {t['description']}" if t.get("description") else ""))
            if action.get("possible_duplicates"):
                st.caption("A transaction with the same amount, category and date already exists.")
            with st.container(horizontal=True):
                st.button("Confirm", key=f"confirm_{action['id']}", type="primary", icon=":material/check:",
                          on_click=resolve_action, args=(action, "confirm"))
                st.button("Cancel", key=f"cancel_{action['id']}", icon=":material/close:",
                          on_click=resolve_action, args=(action, "cancel"))


# --- transactions ---

elif view == "Transactions":
    st.title("Transactions")
    with st.expander("Add transaction", icon=":material/add:", expanded=False):
        tx_type = st.segmented_control("Type", ["expense", "income"], default="expense", key="tx_type")
        with st.form("add_tx", clear_on_submit=True):
            cols = st.columns(3)
            amount = cols[0].number_input(f"Amount ({CURRENCY})", min_value=0.01, step=10.0)
            category = cols[1].selectbox(
                "Category", INCOME_CATEGORIES if tx_type == "income" else EXPENSE_CATEGORIES
            )
            when = cols[2].date_input("Date", value=date.today(), max_value=date.today())
            description = st.text_input("Description")
            if st.form_submit_button("Add", type="primary"):
                resp = api("POST", "/transactions", json={
                    "amount": amount, "type": tx_type or "expense", "category": category,
                    "description": description, "date": when.isoformat(),
                })
                if resp.is_success:
                    st.toast("Transaction added", icon=":material/check:")
                else:
                    st.error(error_detail(resp))

    filters = st.container(horizontal=True, vertical_alignment="bottom")
    period_label = filters.selectbox("Period", list(PERIODS), key="tx_period")
    category_filter = filters.selectbox("Category", ["All"] + EXPENSE_CATEGORIES + INCOME_CATEGORIES)
    search = filters.text_input("Search descriptions")
    params: dict[str, Any] = {"period": PERIODS[period_label], "limit": 500}
    if category_filter != "All":
        params["category"] = category_filter
    if search:
        params["q"] = search

    resp = api("GET", "/transactions", params=params)
    rows = resp.json() if resp.is_success else []
    if not rows:
        st.info("No transactions for these filters.")
    else:
        df = pd.DataFrame(rows)[["date", "type", "category", "amount", "description", "id"]]
        selection = st.dataframe(
            df,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            column_config={
                "amount": st.column_config.NumberColumn("Amount", format=f"{CURRENCY}%.2f"),
                "date": st.column_config.DateColumn("Date"),
                "id": None,
            },
            key="tx_table",
        )
        chosen = selection.selection.rows
        if chosen and st.button(f"Delete {len(chosen)} selected", icon=":material/delete:"):
            for idx in chosen:
                api("DELETE", f"/transactions/{df.iloc[idx]['id']}")
            st.rerun()


# --- dashboard ---

elif view == "Dashboard":
    st.title("Dashboard")
    period_label = st.segmented_control("Period", list(PERIODS), default="This month") or "This month"
    resp = api("GET", "/summary", params={"period": PERIODS[period_label]})
    if not resp.is_success:
        st.error(error_detail(resp))
        st.stop()
    summary = resp.json()

    with st.container(horizontal=True):
        st.metric("Spent", money(summary["total_expense"]), border=True)
        st.metric("Income", money(summary["total_income"]), border=True)
        st.metric("Net", money(summary["net"]), border=True)
        st.metric("Transactions", summary["transaction_count"], border=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Monthly trend")
        trend = api("GET", "/analytics/trend", params={"months": 6}).json()["months"]
        trend_df = pd.DataFrame(trend).rename(columns={"month": "Month", "expense": "Spent", "income": "Income"})
        st.line_chart(trend_df, x="Month", y=["Spent", "Income"], alt="Monthly income and spending, last 6 months")
    with right:
        st.subheader("Spending by category")
        if summary["by_category"]:
            df = pd.DataFrame(summary["by_category"].items(), columns=["Category", "Amount"])
            st.bar_chart(df, x="Category", y="Amount", horizontal=True, sort="-Amount",
                         alt="Spending by category for the selected period")
        else:
            st.info("No expenses in this period yet.")

    budgets = api("GET", "/budgets/status").json()
    if budgets:
        st.subheader("Budgets this month")
        for b in budgets:
            label = f"{b['category']}: {money(b['spent'])} of {money(b['limit'])} ({b['used_pct']:.0f}%)"
            if b["state"] != "ok":
                label += " · over budget" if b["state"] == "over" else " · nearly used"
            st.progress(min(b["used_pct"] / 100, 1.0), text=label)

    left, right = st.columns(2)
    with left:
        st.subheader("Recurring payments")
        recurring = api("GET", "/analytics/recurring").json()
        if recurring["recurring"]:
            st.caption(f"About {money(recurring['estimated_monthly_total'])} per month")
            st.dataframe(
                pd.DataFrame(recurring["recurring"])[["description", "typical_amount", "next_expected"]],
                hide_index=True,
                column_config={
                    "description": "Payee",
                    "typical_amount": st.column_config.NumberColumn("Amount", format=f"{CURRENCY}%.2f"),
                    "next_expected": st.column_config.DateColumn("Next expected"),
                },
                alt="Detected recurring payments",
            )
        else:
            st.caption("Needs at least three months of history.")
    with right:
        st.subheader("Unusual spending")
        unusual = api("GET", "/analytics/unusual", params={"period": "last_30_days"}).json()["unusual"]
        if unusual:
            for t in unusual:
                st.markdown(f"**{money(t['amount'])}** · {t['description'] or t['category']} · {t['date']}")
                st.caption(t["reason"])
        else:
            st.caption("Nothing unusual in the last 30 days.")


# --- budgets & goals ---

elif view == "Budgets & goals":
    st.title("Budgets & goals")
    budget_col, goal_col = st.columns(2)

    with budget_col:
        st.subheader("Monthly budgets")
        with st.form("set_budget", clear_on_submit=True):
            row = st.container(horizontal=True, vertical_alignment="bottom")
            category = row.selectbox("Category", EXPENSE_CATEGORIES)
            limit = row.number_input(f"Monthly limit ({CURRENCY})", min_value=1.0, step=500.0, value=5000.0)
            if row.form_submit_button("Save", type="primary"):
                resp = api("PUT", f"/budgets/{category}", json={"monthly_limit": limit})
                st.toast("Budget saved" if resp.is_success else error_detail(resp))
        for b in api("GET", "/budgets/status").json():
            with st.container(border=True):
                st.progress(min(b["used_pct"] / 100, 1.0),
                            text=f"**{b['category']}** · {money(b['spent'])} of {money(b['limit'])}")
                projected = b.get("projected_month_end")
                caption = f"{money(b['remaining'])} left" if b["remaining"] >= 0 else \
                    f"{money(-b['remaining'])} over"
                if projected is not None:
                    caption += f" · on pace for {money(projected)}"
                st.caption(caption)
                if st.button("Remove", key=f"del_budget_{b['category']}", icon=":material/delete:"):
                    api("DELETE", f"/budgets/{b['category']}")
                    st.rerun()

    with goal_col:
        st.subheader("Savings goals")
        with st.expander("New goal", icon=":material/add:"):
            with st.form("new_goal", clear_on_submit=True):
                name = st.text_input("Name", placeholder="Goa trip")
                target = st.number_input(f"Target ({CURRENCY})", min_value=1.0, step=1000.0, value=50000.0)
                deadline = st.date_input("Target date (optional)", value=None, min_value=date.today())
                if st.form_submit_button("Create", type="primary"):
                    resp = api("POST", "/goals", json={
                        "name": name or "Goal", "target_amount": target,
                        "target_date": deadline.isoformat() if deadline else None,
                    })
                    st.toast("Goal created" if resp.is_success else error_detail(resp))
        for g in api("GET", "/goals").json():
            with st.container(border=True):
                st.progress(g["progress_pct"] / 100,
                            text=f"**{g['name']}** · {money(g['saved_amount'])} of {money(g['target_amount'])}")
                details = []
                if g["completed"]:
                    details.append("Completed")
                elif g["required_per_month"] is not None:
                    details.append(f"Needs {money(g['required_per_month'])}/month by {g['target_date']} · "
                                   + ("on track" if g["on_track"] else "behind"))
                if g["projected_completion"] and not g["completed"]:
                    details.append(f"At your current rate: {g['projected_completion']}")
                if details:
                    st.caption(" · ".join(details))
                row = st.container(horizontal=True, vertical_alignment="bottom")
                amount = row.number_input("Amount", step=500.0, value=1000.0, key=f"amt_{g['id']}",
                                          label_visibility="collapsed")
                if row.button("Add savings", key=f"add_{g['id']}"):
                    api("POST", f"/goals/{g['id']}/contributions", json={"amount": amount})
                    st.rerun()
                if row.button("Delete", key=f"del_{g['id']}", icon=":material/delete:"):
                    api("DELETE", f"/goals/{g['id']}")
                    st.rerun()


# --- import ---

else:
    st.title("Import bank statement")
    st.caption("Upload a CSV export from your bank. Review the categories, then import. Nothing is saved until "
               "you press Import.")
    upload = st.file_uploader("Statement (.csv)", type=["csv"])
    if upload is not None and st.session_state.get("import_file") != upload.file_id:
        resp = api("POST", "/imports", files={"file": (upload.name, upload.getvalue(), "text/csv")})
        st.session_state.import_file = upload.file_id
        st.session_state.import_preview = resp.json() if resp.is_success else None
        if not resp.is_success:
            st.error(error_detail(resp))

    preview = st.session_state.get("import_preview")
    if preview and preview["status"] == "pending":
        s = preview["summary"]
        with st.container(horizontal=True):
            st.metric("Rows", s["rows"], border=True)
            st.metric("Duplicates", s["duplicates"], border=True)
            st.metric("Expenses", money(s["expense_total"]), border=True)
            st.metric("Income", money(s["income_total"]), border=True)
        by = s["categorized_by"]
        st.caption(f"Categorised by rules: {by['rule']} · by AI: {by['llm']} · uncategorised: {by['default']}")
        for err in preview["errors"][:5]:
            st.warning(f"Line {err['line']}: {err['error']}")

        original = pd.DataFrame(preview["rows"])
        edited = st.data_editor(
            original[["include", "date", "description", "amount", "type", "category", "duplicate"]],
            hide_index=True,
            disabled=["date", "description", "amount", "type", "duplicate"],
            column_config={
                "include": st.column_config.CheckboxColumn("Import"),
                "amount": st.column_config.NumberColumn("Amount", format=f"{CURRENCY}%.2f"),
                "category": st.column_config.SelectboxColumn(
                    "Category", options=EXPENSE_CATEGORIES + INCOME_CATEGORIES, required=True
                ),
                "duplicate": st.column_config.CheckboxColumn("Duplicate?"),
            },
            key=f"import_editor_{preview['id']}",
        )
        with st.container(horizontal=True):
            if st.button(f"Import {int(edited['include'].sum())} transactions", type="primary"):
                changes = [
                    {"index": i, "include": bool(edited.at[i, "include"]), "category": edited.at[i, "category"]}
                    for i in range(len(edited))
                    if edited.at[i, "include"] != original.at[i, "include"]
                    or edited.at[i, "category"] != original.at[i, "category"]
                ]
                resp = api("POST", f"/imports/{preview['id']}/commit", json={"rows": changes})
                if resp.is_success:
                    st.success(f"Imported {resp.json()['imported']} transactions.")
                    st.session_state.import_preview = None
                else:
                    st.error(error_detail(resp))
            if st.button("Discard"):
                api("DELETE", f"/imports/{preview['id']}")
                st.session_state.import_preview = None
                st.rerun()
