from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from app.database import Events, Microphones, Nodes

router = APIRouter(prefix="/insights", tags=["insights"])


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse Mongo datetime or common ISO string timestamps."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _to_iso_string(value: Any) -> Optional[str]:
    """
    Format a timestamp for JSON responses without crashing.

    Supports datetime values and ISO / Zulu strings already stored in Mongo.
    Unsupported values return None instead of raising AttributeError.
    """
    if value is None:
        return None

    parsed = _parse_ts(value)
    if parsed is not None:
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return parsed.isoformat()

    if isinstance(value, str) and value.strip():
        return value.strip()

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


def _normalize_timestamp_stage() -> Dict[str, Any]:
    """Convert mixed Mongo timestamp storage (Date or string) into a Date field."""
    return {
        "$addFields": {
            "_normalizedTs": {
                "$switch": {
                    "branches": [
                        {
                            "case": {"$eq": [{"$type": "$timestamp"}, "date"]},
                            "then": "$timestamp",
                        },
                        {
                            "case": {"$eq": [{"$type": "$timestamp"}, "string"]},
                            "then": {
                                "$dateFromString": {
                                    "dateString": "$timestamp",
                                    "onError": None,
                                    "onNull": None,
                                }
                            },
                        },
                    ],
                    "default": None,
                }
            }
        }
    }


def _build_insights_match(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    species: Optional[str] = None,
    sensor_id: Optional[str] = None,
) -> Dict[str, Any]:
    match: Dict[str, Any] = {
        # Skip records whose timestamp could not be normalized.
        "_normalizedTs": {"$ne": None},
    }

    if start is not None or end is not None:
        ts_filter: Dict[str, Any] = {"$ne": None}
        if start is not None:
            ts_filter["$gte"] = start
        if end is not None:
            ts_filter["$lte"] = end
        match["_normalizedTs"] = ts_filter

    if species:
        match["species"] = species

    if sensor_id:
        match["sensorId"] = sensor_id

    return match


@router.get("/overview")
def insights_overview(
    start: Annotated[Optional[str], Query(description="Inclusive start timestamp (ISO 8601)")] = None,
    end: Annotated[Optional[str], Query(description="Inclusive end timestamp (ISO 8601)")] = None,
    species: Annotated[Optional[str], Query(description="Filter by species name (exact match)")] = None,
    sensorId: Annotated[Optional[str], Query(description="Filter by sensor ID (exact match)")] = None,
):
    start_dt = _parse_query_ts(start, "start")
    end_dt = _parse_query_ts(end, "end")

    microphones = Microphones.count_documents({})
    nodes = Nodes.count_documents({})

    match = _build_insights_match(
        start=start_dt,
        end=end_dt,
        species=species,
        sensor_id=sensorId,
    )

    pipeline: List[Dict[str, Any]] = [
        _normalize_timestamp_stage(),
        {"$match": match},
        {
            "$group": {
                "_id": None,
                "detections": {"$sum": 1},
                "species": {"$addToSet": "$species"},
                "sensors": {"$addToSet": "$sensorId"},
                "minTs": {"$min": "$_normalizedTs"},
                "maxTs": {"$max": "$_normalizedTs"},
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

    return {
        "timeRange": {
            "start": _to_iso_string(row.get("minTs")),
            "end": _to_iso_string(row.get("maxTs")),
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
    start: Annotated[Optional[str], Query(description="Inclusive start timestamp (ISO 8601)")] = None,
    end: Annotated[Optional[str], Query(description="Inclusive end timestamp (ISO 8601)")] = None,
    species: Annotated[Optional[str], Query(description="Filter by species name (exact match)")] = None,
    sensorId: Annotated[Optional[str], Query(description="Filter by sensor ID (exact match)")] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
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
        _normalize_timestamp_stage(),
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
