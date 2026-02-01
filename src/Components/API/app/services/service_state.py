from datetime import datetime, timezone
from app.database import ServiceStates


def get_service_state(service: str) -> bool:
    doc = ServiceStates.find_one({"service": service})
    if not doc:
        return False
    return bool(doc.get("paused", False))


def set_service_state(service: str, paused: bool) -> dict:
    now = datetime.now(timezone.utc)
    ServiceStates.update_one(
        {"service": service},
        {"$set": {"service": service, "paused": paused, "updated_at": now}},
        upsert=True,
    )
    return {"service": service, "paused": paused, "updated_at": now}
