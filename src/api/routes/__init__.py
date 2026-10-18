from fastapi import APIRouter

from . import auth, chat, transactions

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(transactions.router)
api_router.include_router(chat.router)
