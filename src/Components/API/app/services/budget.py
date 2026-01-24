from datetime import datetime, timezone
from fastapi import HTTPException, status
from app.database import AdminBudgets


def _month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


def set_budget_limits(rules: list[dict]) -> dict:
    now = datetime.now(timezone.utc)
    AdminBudgets.update_one(
        {"_type": "config"},
        {"$set": {"_type": "config", "rules": rules, "updated_at": now}},
        upsert=True,
    )
    return {"updated_at": now, "rules": rules}


def get_budget_limits() -> dict:
    doc = AdminBudgets.find_one({"_type": "config"}) or {}
    return {"rules": doc.get("rules", []), "updated_at": doc.get("updated_at")}


def _get_limit_for(service: str) -> int:
    config = AdminBudgets.find_one({"_type": "config"}) or {}
    rules = config.get("rules", [])
    for r in rules:
        if r.get("service") == service:
            return int(r.get("monthly_limit", 0))
    return 0


def get_usage(service: str, when: datetime | None = None) -> dict:
    when = when or datetime.now(timezone.utc)
    mk = _month_key(when)

    limit = _get_limit_for(service)
    usage_doc = AdminBudgets.find_one({"_type": "usage", "service": service, "month_key": mk}) or {}
    used = int(usage_doc.get("used", 0))
    remaining = max(limit - used, 0)

    return {"service": service, "monthly_limit": limit, "month_key": mk, "used": used, "remaining": remaining}


def enforce_and_consume(service: str, cost: int = 1):
    now = datetime.now(timezone.utc)
    mk = _month_key(now)
    limit = _get_limit_for(service)

    if limit <= 0:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Budget is not configured for '{service}' (monthly_limit=0).",
        )

    usage_doc = AdminBudgets.find_one({"_type": "usage", "service": service, "month_key": mk}) or {}
    used = int(usage_doc.get("used", 0))

    if used + cost > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Budget exceeded for '{service}'. Used {used}/{limit} this month.",
        )

    AdminBudgets.update_one(
        {"_type": "usage", "service": service, "month_key": mk},
        {"$set": {"_type": "usage", "service": service, "month_key": mk, "updated_at": now},
         "$inc": {"used": cost}},
        upsert=True,
    )
