## app.middleware.admin_auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.middleware.auth import decodeJWT

ADMIN_ROLE="ROLE_ADMIN"
bearer_scheme=HTTPBearer(auto_error=False)


def require_admin(credentials: HTTPAuthorizationCredentials=Depends(bearer_scheme)) -> dict:
    """Allow the request only if the JWT carries the administrator role."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload=decodeJWT(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    if ADMIN_ROLE not in payload.get("roles", []):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator privileges required")
    return payload


def not_in_production():
    """Hide debug/schema endpoints when running in production."""
    if settings.environment == "production":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
