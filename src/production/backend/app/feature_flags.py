"""Reusable Backend feature-flag helpers for C12."""

from typing import Optional

from fastapi import HTTPException, status

from app.config import settings


REALTIME_STREAMING_FLAG = "realtime_streaming_enabled"
ANALYTICS_EXTENSIONS_FLAG = "analytics_extensions_enabled"


class UnknownFeatureFlagError(RuntimeError):
    """Raised when code references a flag that is not defined in Settings."""


def is_feature_enabled(flag_name: str) -> bool:
    """
    Return the current state of a configured feature flag.

    Feature flags must be declared on app.config.Settings. Unknown flags
    fail loudly so typos cannot silently enable or disable behaviour.
    """
    if not hasattr(settings, flag_name):
        raise UnknownFeatureFlagError(
            f"Unknown feature flag: {flag_name}"
        )

    value = getattr(settings, flag_name)

    if not isinstance(value, bool):
        raise UnknownFeatureFlagError(
            f"Feature flag '{flag_name}' is not boolean"
        )

    return value


def require_feature(
    flag_name: str,
    display_name: Optional[str] = None,
) -> None:
    """
    Require an HTTP feature to be enabled.

    Disabled features return a predictable 404 response instead of exposing
    partially integrated Sprint 2 behaviour.
    """
    if is_feature_enabled(flag_name):
        return

    name = display_name or flag_name

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"{name} is disabled",
    )