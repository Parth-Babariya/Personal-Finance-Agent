"""
Streamlit front-end for the Personal Finance Agent.

This module builds a three-tab web UI that communicates with the FastAPI
backend over HTTP.  It demonstrates a clean separation between the
**presentation layer** (this file) and the **business logic** (API + services).

Architecture Notes:
    - The UI is a *thin client* — it contains no financial logic.  All
      computations (summaries, insights) are performed server-side, keeping
      the front-end easy to replace or extend.
    - ``httpx`` is used instead of ``requests`` for its async support and
      timeout handling.
    - ``pandas`` is used only for charting convenience; data manipulation
      still happens in the backend.
"""

import streamlit as st
import httpx
import pandas as pd

# Base URL of the FastAPI backend (must be started separately)
API_BASE = "http://localhost:8003/api/v1"

# --- Page configuration ---
st.set_page_config(page_title="Personal Finance Agent", layout="wide")
st.title("💰 Personal Finance Agent")

if "auth_token" not in st.session_state:
    st.session_state.auth_token = None

st.sidebar.header("Account")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")


def auth_headers() -> dict[str, str]:
    """Build auth header for protected endpoints."""
    if not st.session_state.auth_token:
        return {}
    return {"Authorization": f"Bearer {st.session_state.auth_token}"}


if st.sidebar.button("Sign up"):
    try:
        resp = httpx.post(
            f"{API_BASE}/auth/signup",
            json={"username": username, "password": password},
            timeout=30.0,
        )
        resp.raise_for_status()
        st.session_state.auth_token = resp.json()["access_token"]
        st.sidebar.success("Signed up and logged in.")
    except httpx.HTTPError as exc:
        st.sidebar.error(f"Signup failed: {exc}")

if st.sidebar.button("Login"):
    try:
        resp = httpx.post(
            f"{API_BASE}/auth/login",
            json={"username": username, "password": password},
            timeout=30.0,
        )
        resp.raise_for_status()
        st.session_state.auth_token = resp.json()["access_token"]
        st.sidebar.success("Logged in.")
    except httpx.HTTPError as exc:
        st.sidebar.error(f"Login failed: {exc}")

if st.sidebar.button("Logout"):
    st.session_state.auth_token = None
    st.sidebar.info("Logged out.")

# Create three top-level tabs for the main user workflows
tab_chat, tab_transactions, tab_dashboard = st.tabs(["Chat", "Add Transaction", "Dashboard"])

# ──────────────────────────────────────────────────────────────────────
# Tab 1: Chat — natural-language interaction with the finance agent
# ──────────────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Chat with your Finance Agent")
    message = st.text_input("Ask about your finances")
    if st.button("Send"):
        if message.strip():
            if not st.session_state.auth_token:
                st.warning("Please sign up or login first.")
                st.stop()
            try:
                # POST the user's message to the agent chat endpoint
                resp = httpx.post(
                    f"{API_BASE}/chat",
                    json={"message": message},
                    headers=auth_headers(),
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()
                # Display the agent's reply
                st.write(data["reply"])
                # If the agent returned insights, show each one as an info box
                if data.get("insights"):
                    st.subheader("Insights")
                    for insight in data["insights"]:
                        st.info(insight)
            except httpx.HTTPError as exc:
                st.error(f"API error: {exc}")

# ──────────────────────────────────────────────────────────────────────
# Tab 2: Add Transaction — simple form for manual transaction entry
# ──────────────────────────────────────────────────────────────────────
with tab_transactions:
    st.subheader("Add Transaction")
    amount = st.number_input("Amount ($)", min_value=0.01, step=0.01)
    category = st.selectbox("Category", ["food", "transport", "entertainment", "utilities", "other"])
    description = st.text_input("Description")
    if st.button("Add"):
        if not st.session_state.auth_token:
            st.warning("Please sign up or login first.")
            st.stop()
        try:
            # POST the new transaction to the backend for persistence
            resp = httpx.post(
                f"{API_BASE}/transactions",
                json={"amount": amount, "category": category, "description": description},
                headers=auth_headers(),
                timeout=30.0,
            )
            resp.raise_for_status()
            st.success("Transaction added!")
        except httpx.HTTPError as exc:
            st.error(f"API error: {exc}")

# ──────────────────────────────────────────────────────────────────────
# Tab 3: Dashboard — visual overview of spending data
# ──────────────────────────────────────────────────────────────────────
with tab_dashboard:
    st.subheader("Spending Dashboard")
    if not st.session_state.auth_token:
        st.warning("Please sign up or login first.")
        st.stop()
    try:
        # GET the spending summary from the backend
        resp = httpx.get(f"{API_BASE}/summary", headers=auth_headers(), timeout=30.0)
        resp.raise_for_status()
        summary = resp.json()
        # Display key metrics side-by-side using Streamlit columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Spending", f"${summary['total_spending']:.2f}")
        with col2:
            st.metric("Transactions", summary["transaction_count"])
        # Render a bar chart of spending by category (if data exists)
        if summary.get("by_category"):
            df = pd.DataFrame(
                list(summary["by_category"].items()),
                columns=["Category", "Amount"],
            )
            st.bar_chart(df.set_index("Category"))
    except httpx.HTTPError:
        st.warning("API not running. Start the server first.")
