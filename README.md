# Personal Finance Agent

[![CI](https://github.com/Parth-Babariya/Personal-Finance-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Parth-Babariya/Personal-Finance-Agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

An LLM tool-calling agent for personal finance. Users track income and expenses through a REST API or a
Streamlit UI, then ask questions in plain language ("did I spend more on food than last month?", "I spent
450 on lunch"). The agent decides which tools to call, queries only that user's data, and answers with
numbers taken from the database.

## Highlights

- **Tool-calling agent loop.** The LLM picks from 6 typed tools, sees their results, and writes the answer.
  It can call several tools in one step and supports multi-step reasoning with a step budget.
- **Guardrails.** Writes need user confirmation, money figures in replies are checked against tool
  results at runtime, card/Aadhaar/PAN numbers are masked before reaching the LLM, and rate limits plus a
  daily token budget cap abuse and cost. Details in [Guardrails](#guardrails).
- **Streaming.** `POST /chat/stream` sends Server-Sent Events: tool calls as they happen, then answer tokens.
- **Multi-turn memory.** Conversation history is stored in SQL per user, so follow-ups like "and last
  month?" work.
- **Works without an LLM.** A deterministic keyword planner uses the same tools when no API key is set or
  the provider fails. It never re-runs a write the LLM already made.
- **Any OpenAI-compatible provider.** OpenAI, Groq, Gemini or Ollama, selected with `OPENAI_BASE_URL`.
- **Eval harness.** 34 cases (summaries, comparisons, lookups, writes, multi-turn, guardrails). Each
  reply is checked against ground truth computed from the database. See [evals/RESULTS.md](evals/RESULTS.md).
- **Solid basics.** Money stored as `Numeric(12,2)`, income and expense types, real date-range periods,
  full transaction CRUD with filters, Alembic migrations, 156 tests on an isolated in-memory
  database, and lint plus migration checks in CI.

## Architecture

```mermaid
flowchart LR
    UI[Streamlit UI] -->|REST + SSE| API[FastAPI routes]
    API --> Auth[JWT auth]
    API --> Agent[FinanceAgent]
    Agent -->|messages + tool schemas| LLM[(OpenAI-compatible LLM)]
    Agent -. no key / provider error .-> FB[Keyword fallback planner]
    Agent --> Reg[ToolRegistry]
    FB --> Reg
    Reg --> Tx[SQLTransactionService]
    Agent --> Conv[ConversationStore]
    Tx --> DB[(SQLite / Postgres)]
    Conv --> DB
```

One chat turn:

1. Load the user's last N messages from `chat_messages`.
2. Send the system prompt (today's date, currency, rules), the history, the user's message and the tool
   schemas to the LLM.
3. If the model requests tools, validate the arguments, run them in a worker thread scoped to the
   authenticated user, append the results, and call the model again. Stop at a text answer or after
   `AGENT_MAX_STEPS`; when the budget runs out, the model must answer without tools.
4. Save the user message and the reply (with the names of the tools used), and emit a final `done` event
   with the tool trace, latency and token usage.

### Tools

| Tool | Purpose |
|---|---|
| `get_spending_summary` | Expenses, income, net and per-category totals for a period or date range |
| `compare_periods` | Two periods side by side, with the change overall and per category |
| `search_transactions` | Filter by text, category, type, amount range and dates |
| `get_top_expenses` | Largest expenses in a period |
| `get_insights` | Rule-based observations (top category, large purchases, savings rate) |
| `add_transaction` | Propose an expense or income from natural language; saved only after the user confirms |

Periods: `today`, `this_week`, `last_7_days`, `this_month`, `last_month`, `last_30_days`, `this_year`,
`last_year`, `all_time`, or explicit `start_date`/`end_date`.

## Guardrails

| Risk | Guardrail |
|---|---|
| Model reads another user's data | `user_id` comes from the JWT via `ToolContext`; it is not a tool argument, so no prompt can change it |
| Malformed or out-of-range tool calls | Pydantic validation (enums, ranges, no future dates, no extra fields); errors go back to the model |
| Misread message adds a wrong transaction | `add_transaction` creates a pending action (15 min expiry); saved only after "yes" or the Confirm button; possible duplicates flagged; max 3 writes per message |
| Invented numbers | After each LLM reply, every currency figure must match a tool result, the conversation, or a sum/difference of tool numbers; one corrective retry, then a visible warning |
| Sensitive identifiers sent to the provider | Card numbers (Luhn-checked), Aadhaar, PAN, phone, email/UPI and account numbers masked before the LLM call and before storage |
| Runaway loops or slow providers | Step budget with a forced final answer, max tool calls per step, and an overall deadline (`AGENT_TIMEOUT_SECONDS`) with fallback |
| Cost abuse and brute force | Per-user chat rate limit, per-IP auth limit, lockout after repeated failed logins, daily LLM token budget per user (then fallback) |
| Off-topic requests and advice | System prompt scopes the assistant to budgeting and declines investment, tax or legal advice; covered by eval cases |

The rate limiter is in-process; with several API replicas it should move to a shared store such as Redis.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # source .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt
copy .env.example .env                               # then edit; or export the variables
uvicorn src.api.app:app --port 8003 --reload         # API + docs at http://localhost:8003/docs
streamlit run src/ui/app.py                          # UI at http://localhost:8501
```

Leave `OPENAI_API_KEY` empty to run with the keyword fallback. To use a free-tier provider instead of
OpenAI, set `OPENAI_BASE_URL` and `OPENAI_MODEL`; examples are in [.env.example](.env.example).

For production, set `APP_ENV=production` (the app then refuses to start without `JWT_SECRET_KEY`), point
`DATABASE_URL` at Postgres, set `AUTO_CREATE_TABLES=false`, and run `alembic upgrade head`.

## API

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/auth/signup`, `/api/v1/auth/login` | Get a JWT |
| POST | `/api/v1/transactions` | Add an expense or income |
| GET | `/api/v1/transactions` | List, with `period`/`start_date`/`end_date`/`category`/`type`/`q`/pagination |
| GET / PATCH / DELETE | `/api/v1/transactions/{id}` | Read, partially update, delete |
| GET | `/api/v1/summary` | Totals for a period or date range |
| POST | `/api/v1/chat` | Agent reply, plus the tool trace, mode, pending actions and grounding report |
| POST | `/api/v1/chat/stream` | The same as SSE events: `tool_call`, `tool_result`, `token`, `reset`, `done` |
| GET | `/api/v1/actions` | Open proposals waiting for confirmation |
| POST | `/api/v1/actions/{id}/confirm`, `/cancel` | Apply or discard a proposed write |
| GET / DELETE | `/api/v1/chat/history` | View or clear conversation history |
| GET | `/api/v1/memory/search` | Keyword search over past conversation |
| GET | `/health` | Liveness check and whether the LLM is enabled |

## Testing and evaluation

```bash
pytest --cov                                 # unit and API tests (in-memory DB, fake LLM, mocked HTTP)
ruff check src tests evals migrations
python -m evals.run_eval --mode fallback     # no API key needed
python -m evals.run_eval --mode llm -v       # real model; checks the provider responds first
python -m evals.run_eval --mode both --write-results   # regenerate evals/RESULTS.md
```

Every eval case runs on a fresh database seeded with the same 3.5 months of synthetic transactions and a
fixed "today", so results are reproducible. A case passes only if the expected tools were called,
forbidden tools were not (for example, no `add_transaction` for "I'm thinking about buying a laptop"),
and every expected number appears in the reply.

## Project structure

```text
src/
  agent/       agent.py (loop + streaming), tools.py (typed tools + registry),
               llm.py (provider client), fallback.py (keyword planner), prompts.py
  api/         app.py (factory), routes/ (auth, transactions, chat), schemas.py, deps.py
  services/    SQL transaction service, conversation store, auth, periods, insights, models
  interfaces/  abstract contracts the agent and routes depend on
  ui/          Streamlit client
evals/         dataset, cases with DB-derived ground truth, runner
migrations/    Alembic
tests/
```
