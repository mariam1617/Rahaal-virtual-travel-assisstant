"""
FastAPI Router for User Authentication (Sign In & Login)
Module: src/api/auth.py
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.api.models import (
    RegisterRequest,
    LoginRequest,
    AuthResponse,
    UserResponse,
)
from src.services.auth import (
    AuthService,
    decode_access_token,
    create_access_token,
)


router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)

# Singleton service instance
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Dependency provider for AuthService."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(
    request: RegisterRequest,
    service: AuthService = Depends(get_auth_service)
) -> AuthResponse:
    """
    Register a new traveler account.
    Returns authenticated user information and a signed JWT access token.
    """
    success, message, user = service.register(
        username=request.username,
        password=request.password,
        name=request.name,
        email=request.email,
        nationality=request.nationality,
    )

    if not success or not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )

    # Issue access token immediately for seamless onboarding
    token = create_access_token({
        "sub": user.user_id,
        "username": user.username,
        "name": user.name,
        "nationality": user.nationality,
    })

    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user_id=user.user_id,
        username=user.username,
        name=user.name,
        nationality=user.nationality,
        message=message,
    )


@router.post("/login", response_model=AuthResponse)
def login(
    request: LoginRequest,
    service: AuthService = Depends(get_auth_service)
) -> AuthResponse:
    """
    Authenticate an existing user with username and password.
    Returns authenticated user information and a signed JWT access token.
    """
    success, message, user, token = service.login(
        username=request.username,
        password=request.password,
    )

    if not success or not user or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message
        )

    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user_id=user.user_id,
        username=user.username,
        name=user.name,
        nationality=user.nationality,
        message=message,
    )


@router.get("/me", response_model=UserResponse)
def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    service: AuthService = Depends(get_auth_service)
) -> UserResponse:
    """
    Fetches the authenticated traveler's profile from the JWT bearer token.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header."
        )

    payload = decode_access_token(credentials.credentials)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token."
        )

    user_id = payload["sub"]
    user = service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User account not found."
        )

    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        name=user.name,
        email=user.email,
        nationality=user.nationality,
        created_at=user.created_at,
    )
