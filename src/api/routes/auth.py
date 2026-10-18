"""Signup and login endpoints, rate limited per client IP and per username."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from ...config import Settings
from ...services.auth_service import AuthService
from ...services.security import create_access_token
from ..deps import get_auth_service, get_settings, rate_limit_auth, too_many_requests
from ..schemas import TokenResponse, UserCredentials

router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(rate_limit_auth)])


def _token(user_id: str, settings: Settings) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user_id, settings.jwt_secret_key, settings.jwt_expire_minutes)
    )


def _authenticate(request: Request, auth_service: AuthService, username: str, password: str) -> str:
    """Check credentials, locking a username out after repeated failures."""
    failures = request.app.state.limiters.login_failures
    key = username.lower()
    wait = failures.retry_after(key)
    if wait is not None:
        too_many_requests(wait, "Too many failed login attempts. Try again later.")
    user = auth_service.authenticate_user(username, password)
    if user is None:
        failures.hit(key)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    failures.reset(key)
    return user.id


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
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    return _token(_authenticate(request, auth_service, req.username, req.password), settings)


@router.post("/token", response_model=TokenResponse, include_in_schema=False)
def token(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """OAuth2 form login so the 'Authorize' button in /docs works."""
    return _token(_authenticate(request, auth_service, form.username, form.password), settings)
