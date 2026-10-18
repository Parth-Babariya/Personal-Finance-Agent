from fastapi import APIRouter

from . import actions, analytics, auth, budgets, chat, goals, imports, transactions

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(transactions.router)
api_router.include_router(chat.router)
api_router.include_router(actions.router)
api_router.include_router(budgets.router)
api_router.include_router(analytics.router)
api_router.include_router(goals.router)
api_router.include_router(imports.router)
