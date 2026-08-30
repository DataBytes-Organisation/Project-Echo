from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from app.database import Events, Microphones, Nodes

router = APIRouter(prefix="/insights", tags=["insights"])


def _parse_ts(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _parse_query_ts(value: Optional[str], field_name: str) -> Optional[datetime]:
    if value is None:
        return None

    parsed = _parse_ts(value)
    if parsed is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name} datetime format",
        )
    return parsed


def _build_insights_match(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    species: Optional[str] = None,
    sensor_id: Optional[str] = None,
) -> Dict[str, Any]:
    match: Dict[str, Any] = {}

    if start is not None or end is not None:
        ts_filter: Dict[str, Any] = {}
        if start is not None:
            ts_filter["$gte"] = start
        if end is not None:
            ts_filter["$lte"] = end
        match["timestamp"] = ts_filter

    if species:
        match["species"] = species

    if sensor_id:
        match["sensorId"] = sensor_id

    return match


@router.get("/overview")
def insights_overview(
    start: Optional[str] = Query(None, description="Inclusive start timestamp (ISO 8601)"),
    end: Optional[str] = Query(None, description="Inclusive end timestamp (ISO 8601)"),
    species: Optional[str] = Query(None, description="Filter by species name (exact match)"),
    sensorId: Optional[str] = Query(None, description="Filter by sensor ID (exact match)"),
):
    start_dt = _parse_query_ts(start, "start")
    end_dt = _parse_query_ts(end, "end")
    match = _build_insights_match(
        start=start_dt,
        end=end_dt,
        species=species,
        sensor_id=sensorId,
    )

    microphones = Microphones.count_documents({})
    nodes = Nodes.count_documents({})

    pipeline = [
        {"$match": match},
        {
            "$group": {
                "_id": None,
                "detections": {"$sum": 1},
                "species": {"$addToSet": "$species"},
                "sensors": {"$addToSet": "$sensorId"},
                "minTs": {"$min": "$timestamp"},
                "maxTs": {"$max": "$timestamp"},
            }
        },
    ]

    summary = list(Events.aggregate(pipeline))
    if not summary:
        return {
            "timeRange": {"start": None, "end": None},
            "counts": {
                "detections": 0,
                "uniqueSpecies": 0,
                "sensorsWithDetections": 0,
                "microphones": microphones or nodes,
            },
        }

    row = summary[0]
    species_values = [value for value in row.get("species", []) if value]
    sensor_values = [value for value in row.get("sensors", []) if value]

    min_ts = row.get("minTs")
    max_ts = row.get("maxTs")

    return {
        "timeRange": {
            "start": min_ts.isoformat() if min_ts else None,
            "end": max_ts.isoformat() if max_ts else None,
        },
        "counts": {
            "detections": int(row.get("detections", 0)),
            "uniqueSpecies": len(species_values),
            "sensorsWithDetections": len(sensor_values),
            "microphones": microphones or nodes,
        },
    }


@router.get("/species")
def insights_species(
    start: Optional[str] = Query(None, description="Inclusive start timestamp (ISO 8601)"),
    end: Optional[str] = Query(None, description="Inclusive end timestamp (ISO 8601)"),
    species: Optional[str] = Query(None, description="Filter by species name (exact match)"),
    sensorId: Optional[str] = Query(None, description="Filter by sensor ID (exact match)"),
    limit: int = Query(10, ge=1, le=50),
):
    start_dt = _parse_query_ts(start, "start")
    end_dt = _parse_query_ts(end, "end")
    match = _build_insights_match(
        start=start_dt,
        end=end_dt,
        species=species,
        sensor_id=sensorId,
    )
    if species:
        match["species"] = species
    else:
        match["species"] = {"$exists": True, "$ne": ""}

    pipeline = [
        {"$match": match},
        {
            "$group": {
                "_id": "$species",
                "count": {"$sum": 1},
                "avg_confidence": {"$avg": "$confidence"},
            }
        },
        {"$sort": {"count": -1}},
        {"$limit": limit},
        {"$project": {"_id": 0, "species": "$_id", "count": 1, "avg_confidence": 1}},
    ]

    return {
        "items": list(Events.aggregate(pipeline))
    }
