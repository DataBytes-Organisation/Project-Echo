from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Body, Depends, HTTPException

from app.middleware.pause_guard import pause_guard
from app.services.budget import (
    set_budget_limits,
    get_budget_limits,
    get_usage,
    reset_usage,
)

router = APIRouter(
    prefix="/admin/budget",
    tags=["admin-budget"],
)


@router.get(
    "/limits",
    summary="Get current budget limits config",
    dependencies=[Depends(pause_guard("admin_budget"))],
)
def get_limits_endpoint():
    return get_budget_limits()


@router.post(
    "/limits",
    summary="Set budget limits config (rules)",
    dependencies=[Depends(pause_guard("admin_budget"))],
)
def set_limits_endpoint(
    rules: List[Dict[str, Any]] = Body(..., description="List of budget rules"),
):
    if not isinstance(rules, list) or len(rules) == 0:
        raise HTTPException(status_code=400, detail="rules must be a non-empty list")
    return set_budget_limits(rules)


@router.get(
    "/usage",
    summary="Get current usage for a service (optionally month)",
    dependencies=[Depends(pause_guard("admin_budget"))],
)
def get_usage_endpoint(service: str, month_key: Optional[str] = None):
    return get_usage(service=service, month_key=month_key)


@router.post(
    "/usage/reset",
    summary="Reset usage counter for a service (optionally month)",
    dependencies=[Depends(pause_guard("admin_budget"))],
)
def reset_usage_endpoint(service: str, month_key: Optional[str] = None):
    return reset_usage(service=service, month_key=month_key)

