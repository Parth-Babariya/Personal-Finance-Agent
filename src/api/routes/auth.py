"""Signup and login endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from ...config import Settings
from ...services.auth_service import AuthService
from ...services.security import create_access_token
from ..deps import get_auth_service, get_settings
from ..schemas import TokenResponse, UserCredentials

router = APIRouter(prefix="/auth", tags=["auth"])


def _token(user_id: str, settings: Settings) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user_id, settings.jwt_secret_key, settings.jwt_expire_minutes)
    )


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(
    req: UserCredentials,
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    try:
        user = auth_service.register_user(req.username, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _token(user.id, settings)


@router.post("/login", response_model=TokenResponse)
def login(
    req: UserCredentials,
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    user = auth_service.authenticate_user(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    return _token(user.id, settings)


@router.post("/token", response_model=TokenResponse, include_in_schema=False)
def token(
    form: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """OAuth2 form login so the 'Authorize' button in /docs works."""
    user = auth_service.authenticate_user(form.username, form.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    return _token(user.id, settings)
