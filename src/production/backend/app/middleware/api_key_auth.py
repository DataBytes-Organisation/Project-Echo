import os

from fastapi import Header, HTTPException, status


def require_api_key(x_api_key: str = Header(default=None)):
    """
    FastAPI dependency. Rejects the request with 401 unless x-api-key
    matches the INTERNAL_API_KEY environment variable.
    """
    expected_key = os.getenv("INTERNAL_API_KEY")
    if not expected_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized Engine Access",
        )
    return True
