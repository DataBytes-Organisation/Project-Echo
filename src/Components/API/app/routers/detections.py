from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Body
from app.schemas import DetectionCreate, Detection
from app import detections as detections_service

router = APIRouter(
    prefix="/detections",
    tags=["detections"],
)


@router.post(
    "/",
    response_model=Detection,
    status_code=201,
    summary="Store a new detection",
)
def create_detection_endpoint(payload: DetectionCreate) -> Detection:
    return detections_service.create_detection(payload)


@router.get(
    "/",
    response_model=List[Detection],
    summary="List detections",
)
def list_detections_endpoint(
    sensor_id: Optional[str] = Query(None, alias="sensorId"),
    species: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[Detection]:
    return detections_service.list_detections(
        sensor_id=sensor_id,
        species=species,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{detection_id}",
    response_model=Detection,
    summary="Get detection by ID",
)
def get_detection_endpoint(detection_id: str) -> Detection:
    detection = detections_service.get_detection(detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    return detection


@router.delete(
    "/{detection_id}",
    status_code=204,
    summary="Delete detection by ID",
)
def delete_detection_endpoint(detection_id: str) -> None:
    deleted = detections_service.delete_detection(detection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Detection not found")


@router.patch(
    "/{detection_id}",
    response_model=Detection,
    summary="Update detection by ID",
)
def update_detection_endpoint(
    detection_id: str,
    payload: Dict[str, Any] = Body(...),
) -> Detection:
    updated = detections_service.update_detection(detection_id, payload)
    if not updated:
        raise HTTPException(status_code=404, detail="Detection not found")
    return updated
