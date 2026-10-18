"""Streamlit front-end for the Personal Finance Agent.

A thin client: all finance logic lives in the FastAPI backend. Run with
``streamlit run src/ui/app.py`` after starting the API.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import date
from typing import Any

import httpx
import pandas as pd
import streamlit as st

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
        st.rerun()
    else:
        st.error(error_detail(resp))


def stream_reply(message: str, steps: Any) -> Iterator[str]:
    """Yield answer tokens from the SSE endpoint; show tool calls as timeline steps."""
    with httpx.stream(
        "POST", f"{API_BASE}/chat/stream", json={"message": message},
        headers={"Authorization": f"Bearer {st.session_state.token}"}, timeout=120.0,
    ) as resp:
        if not resp.is_success:
            resp.read()
            yield f"Error: {error_detail(resp)}"
            return
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "token":
                yield event["text"]
            elif event["type"] == "tool_call":
                with steps:
                    with st.status(f"`{event['name']}`", type="step", state="complete"):
                        st.json(event["arguments"], expanded=False)
            elif event["type"] == "done":
                st.session_state.last_done = event
            elif event["type"] == "error":
                yield f"\n\nError: {event['detail']}"


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
    view = st.segmented_control(
        "View", ["Chat", "Transactions", "Dashboard"], default="Chat", label_visibility="collapsed"
    ) or "Chat"
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
            reply = st.write_stream(stream_reply(prompt, steps))
            done = st.session_state.pop("last_done", None)
            if done:
                st.caption(f"{done['mode']} · {done['latency_ms']:.0f} ms")
        st.session_state.messages += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply if isinstance(reply, str) else "".join(reply),
             "meta": {"tools": [t["name"] for t in (done or {}).get("tool_calls", [])]}},
        ]


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

else:
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

    if summary["by_category"]:
        df = pd.DataFrame(summary["by_category"].items(), columns=["Category", "Amount"])
        st.subheader("Spending by category")
        st.bar_chart(df, x="Category", y="Amount", horizontal=True, sort="-Amount",
                     alt="Spending by category for the selected period")
    else:
        st.info("No expenses in this period yet.")
