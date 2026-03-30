# Personal Finance Agent

[![CI](https://github.com/Parth-Babariya/Personal-Finance-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Parth-Babariya/Personal-Finance-Agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)

A modular AI-powered personal finance assistant with a FastAPI backend, SQL persistence, JWT authentication, memory-backed chat context, and LLM-orchestrated tool routing.

## What It Does
This project helps users track spending, ask finance-related questions in natural language, and get clear insights from their transaction history. It combines transaction APIs, chat-based analysis, and dashboard visualization in one workflow.

## High-Level Overview
- User signs up/logs in and gets an authenticated session.
- Transactions are stored with user-level isolation in the database.
- Summary endpoint computes spending totals and category breakdown.
- Chat endpoint uses an orchestrated finance agent that selects tools and returns contextual responses.
- Session memory helps retain context for better follow-up interactions.
- Streamlit UI provides a simple interface for chat, data entry, and dashboard views.

## Architecture Snapshot
```text
Streamlit UI -> FastAPI Routes -> Services -> SQLAlchemy Models

Services:
- AuthService
- SQLTransactionService
- FinanceLLMAgent
- SimpleInsightGenerator
- SessionMemory
```

## Applications
- Personal expense tracking and budgeting assistants.
- Finance chatbots for spending analysis and habit awareness.
- Starter architecture for AI-enabled fintech dashboards.
- Backend interview project demonstrating API design, auth, persistence, testing, and metrics.

## Core Capabilities
- Add and list transactions through REST APIs.
- Generate spending summaries and category-wise breakdown.
- LLM-orchestrated chat endpoint with deterministic fallback when LLM is unavailable.
- Session memory search to recall previous interaction context.
- JWT-based authentication and user-scoped finance data isolation.
- Streamlit dashboard for quick visualization.

## Tech Stack
- Python
- FastAPI + Pydantic
- SQLAlchemy
- Streamlit
- httpx
- pytest + pytest-asyncio

## Quick Start
```bash
pip install -r requirements.txt
set DATABASE_URL=sqlite:///./finance_agent.db
set JWT_SECRET_KEY=replace-with-a-strong-secret
set OPENAI_API_KEY=your-openai-api-key
set OPENAI_MODEL=gpt-4o-mini
uvicorn src.api.app:app --host 0.0.0.0 --port 8003 --reload
streamlit run src/ui/app.py
```

If `OPENAI_API_KEY` is not set, the chat agent uses deterministic fallback routing.

Backend runs at `http://localhost:8003` and UI calls `http://localhost:8003/api/v1`.

## API Snapshot
- `POST /api/v1/auth/signup`
- `POST /api/v1/auth/login`
- `POST /api/v1/transactions`
- `GET /api/v1/transactions`
- `GET /api/v1/summary`
- `POST /api/v1/chat`
- `GET /api/v1/memory/search`

## Testing
```bash
pytest -q
```

Optional quality run:
```bash
pytest -q tests --cov=src
```

## Quality Snapshot
- Automated tests: 42
- Pass rate: 100%
- Line coverage: 80.00%

## Agent Metrics Snapshot
- Eval prompts: 8 (local benchmark set)
- Tool routing accuracy: 100.00%
- Grounded response rate: 100.00%
- Memory-action trigger rate: 25.00%
- Fallback safety rate: 100.00%
- Agent response latency: mean 0.02 ms, p95 0.06 ms

## Note
Default local setup uses SQLite via SQLAlchemy for persistence. You can switch to PostgreSQL by updating `DATABASE_URL`.
