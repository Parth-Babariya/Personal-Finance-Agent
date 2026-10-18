from fastapi import APIRouter

from . import actions, auth, budgets, chat, transactions

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(transactions.router)
api_router.include_router(chat.router)
api_router.include_router(actions.router)
api_router.include_router(budgets.router)
