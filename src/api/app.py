"""FastAPI application for the Personal Finance Agent.

Sets up the application with proper lifecycle management for services.
Services are initialized during app startup (lifespan) and stored in
app.state, making them available to all routes via dependency injection.

This approach avoids module-level singletons, gives explicit control over
service lifecycle, and makes the app easier to test.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routes import router
from ..services.auth_service import AuthService
from ..services.db import SessionLocal, init_db
from ..services.insight_generator import SimpleInsightGenerator
from ..services.llm_agent import FinanceLLMAgent
from ..services.memory_manager import SessionMemory
from ..services.sql_transaction_service import SQLTransactionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared services on startup, clean up on shutdown.

    Using a lifespan context manager is the recommended FastAPI pattern
    for managing service instances that need to live across requests.
    This replaces module-level singletons, giving us:
      - Explicit initialization and cleanup
      - Easy swapping for tests (override app.state)
      - No unbounded memory growth from leaked references
    """
    # Initialize persistent database and shared services.
    init_db()
    app.state.tx_service = SQLTransactionService(SessionLocal)
    app.state.auth_service = AuthService(SessionLocal)
    app.state.memory = SessionMemory()
    app.state.insight_gen = SimpleInsightGenerator()
    app.state.agent = FinanceLLMAgent(
      tx_service=app.state.tx_service,
      memory=app.state.memory,
      insight_gen=app.state.insight_gen,
    )
    yield
    # Cleanup: clear memory on shutdown to release resources
    app.state.memory.clear()


app = FastAPI(
    title="Personal Finance Agent",
    description="Stateful AI agent with memory for personal finance",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")
