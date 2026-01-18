from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, Query, Path, Body, HTTPException, Depends

from app.schemas import Detection, DetectionCreate, DetectionListResponse
from app import detections as detections_service
from app.middleware.pause_guard import pause_guard
from app.services.budget import enforce_and_consume

router = APIRouter(
    prefix="/detections",
    tags=["detections"],
)

@router.post(
    "",
    response_model=Detection,
    summary="Create a new detection",
    dependencies=[Depends(pause_guard("detections"))],
)
def create_detection_endpoint(payload: DetectionCreate = Body(...)):
    enforce_and_consume("detections", cost=1)
    return detections_service.create_detection(payload)


@router.get(
    "",
    response_model=DetectionListResponse,
    summary="List detections with pagination and filtering",
    dependencies=[Depends(pause_guard("detections"))],
)
def list_detections_endpoint(
    species: Optional[str] = Query(None, description="Filter by species name (exact match)"),
    start_time: Optional[datetime] = Query(
        None, description="Filter detections from this timestamp (inclusive, ISO 8601)"
    ),
    end_time: Optional[datetime] = Query(
        None, description="Filter detections up to this timestamp (inclusive, ISO 8601)"
    ),
    lat: Optional[float] = Query(None, description="Latitude for location filter (requires lon & radius_km)"),
    lon: Optional[float] = Query(None, description="Longitude for location filter (requires lat & radius_km)"),
    radius_km: Optional[float] = Query(None, description="Radius in kilometres for location filter (requires lat & lon)"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of detections per page (max 100)"),
):
    enforce_and_consume("detections", cost=1)

    return detections_service.list_detections(
        species=species,
        start_time=start_time,
        end_time=end_time,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{detection_id}",
    response_model=Detection,
    summary="Get a single detection by id",
    dependencies=[Depends(pause_guard("detections"))],
)
def get_detection_endpoint(
    detection_id: str = Path(..., description="MongoDB ObjectId of the detection"),
):
    return detections_service.get_detection(detection_id)


@router.patch(
    "/{detection_id}",
    response_model=Detection,
    summary="Update fields of a detection",
    dependencies=[Depends(pause_guard("detections"))],
)
def update_detection_endpoint(
    detection_id: str = Path(..., description="MongoDB ObjectId of the detection"),
    updates: Dict[str, Any] = Body(..., description="Partial update payload"),
):
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided for update")

    enforce_and_consume("detections", cost=1)

    return detections_service.update_detection(detection_id, updates)


@router.delete(
    "/{detection_id}",
    summary="Delete a detection by id",
    dependencies=[Depends(pause_guard("detections"))],
)
def delete_detection_endpoint(
    detection_id: str = Path(..., description="MongoDB ObjectId of the detection"),
):
    enforce_and_consume("detections", cost=1)

    detections_service.delete_detection(detection_id)
    return {"deleted": True}


@router.post(
    "/predict",
    summary="Run species prediction (budget-controlled)",
    dependencies=[Depends(pause_guard("species_predictor"))],
)
def predict_endpoint(payload: Dict[str, Any] = Body(...)):
    enforce_and_consume("species_predictor", cost=5)

    return {"status": "not_implemented_here", "received": payload}
