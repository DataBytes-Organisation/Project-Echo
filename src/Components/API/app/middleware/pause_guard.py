from fastapi import HTTPException, status
from app.services.service_state import get_service_state


def pause_guard(service_name: str):
    """
    FastAPI dependency. If service is paused -> 503.
    """
    def _guard():
        if get_service_state(service_name):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service '{service_name}' is temporarily paused by admin.",
            )
        return True

    return _guard