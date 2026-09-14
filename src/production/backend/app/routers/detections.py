from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from fastapi import APIRouter, Query, Path, Body, HTTPException, Depends

from app.schemas import Detection, DetectionCreate, DetectionListResponses
from app import detections as detections_service
from app.middleware.pause_guard import pause_guard
from app.services.budget import enforce_and_consume

router = APIRouter(
    prefix="/detections",
    tags=["detections"],
)

def _field(item: Any, name: str, default=None):
    if isinstance(item, dict):
        return item.get(name, default)

    return getattr(item, name, default)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo:
            return value

        return value.replace(tzinfo=timezone.utc)

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except ValueError:
            return None

    return None

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
    response_model=DetectionListResponses,
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
    "/dashboard-summary",
    summary="Detection counts by sensor and day for the dashboard",
    dependencies=[Depends(pause_guard("detections"))],
)
def dashboard_summary_endpoint(
    days: int = Query(8, ge=1, le=30),
):
    end_time = datetime.now(timezone.utc)

    start_day = (
        end_time - timedelta(days=days - 1)
    ).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    detections = []
    page = 1
    page_size = 100

    while True:
        result = detections_service.list_detections(
            species=None,
            start_time=start_day,
            end_time=end_time,
            lat=None,
            lon=None,
            radius_km=None,
            page=page,
            page_size=page_size,
        )

        items = _field(result, "items", []) or []
        detections.extend(items)

        total = int(
            _field(result, "total", len(detections)) or 0
        )

        if not items or len(detections) >= total:
            break

        page += 1

    days_list = [
        start_day + timedelta(days=index)
        for index in range(days)
    ]

    day_keys = [
        day.strftime("%Y-%m-%d")
        for day in days_list
    ]

    labels = [
        day.strftime("%d/%m")
        for day in days_list
    ]

    counts_by_sensor: Dict[str, Dict[str, int]] = {}

    for detection in detections:
        timestamp = _parse_timestamp(
            _field(detection, "timestamp")
        )

        if timestamp is None:
            continue

        sensor_id = str(
            _field(detection, "sensorId", "Unknown")
        )

        date_key = (
            timestamp
            .astimezone(timezone.utc)
            .strftime("%Y-%m-%d")
        )

        if date_key not in day_keys:
            continue

        sensor_counts = counts_by_sensor.setdefault(
            sensor_id,
            {key: 0 for key in day_keys},
        )

        sensor_counts[date_key] += 1

    series = [
        {
            "name": f"Microphone {sensor_id}",
            "data": [
                counts[day_key]
                for day_key in day_keys
            ],
        }
        for sensor_id, counts
        in sorted(counts_by_sensor.items())
    ]

    return {
        "labels": labels,
        "series": series,
        "total": len(detections),
        "start_time": start_day.isoformat(),
        "end_time": end_time.isoformat(),
    }
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
