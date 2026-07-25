from fastapi import APIRouter
from app.schemas import ServicePauseIn, ServicePauseOut
from app.services.service_state import get_service_state, set_service_state
from datetime import datetime

router = APIRouter()

@router.get("/admin/services/{service}/status", response_model=ServicePauseOut)
def get_pause_status(service: str):
    paused = get_service_state(service)
    # if no doc exists, we still return a valid status
    return ServicePauseOut(service=service, paused=paused, updated_at=datetime.utcnow())

@router.post("/admin/services/pause", response_model=ServicePauseOut)
def pause_or_resume_service(payload: ServicePauseIn):
    result = set_service_state(payload.service, payload.paused)
    return ServicePauseOut(**result)

@router.get("/admin/services", response_model=list[ServicePauseOut])
def list_services_status():
    # list everything known in DB
    docs = list(__import__("app.database").database.ServiceStates.find({}, {"_id": 0}))
    out = []
    for d in docs:
        out.append(ServicePauseOut(
            service=d.get("service"),
            paused=bool(d.get("paused", False)),
            updated_at=d.get("updated_at") or datetime.utcnow(),
        ))
    return out
