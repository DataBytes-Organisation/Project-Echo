import os
from fastapi import APIRouter, Query
from datetime import datetime
from typing import Optional
import pymongo

router = APIRouter(prefix="/insights", tags=["insights"])

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "EchoNet")

client = pymongo.MongoClient(MONGO_URI)
db = client[MONGO_DB]


def parse_ts(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


@router.get("/overview")
def insights_overview(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    events = list(db["events"].find())
    microphones = db["microphones"].count_documents({})
    nodes = db["nodes"].count_documents({})

    start_dt = parse_ts(start)
    end_dt = parse_ts(end)

    filtered = []
    for e in events:
        ts = parse_ts(e.get("timestamp"))
        if not ts:
            continue
        if start_dt and ts < start_dt:
            continue
        if end_dt and ts > end_dt:
            continue
        filtered.append(e)

    timestamps = [parse_ts(e["timestamp"]) for e in filtered if parse_ts(e.get("timestamp"))]

    return {
        "timeRange": {
            "start": min(timestamps).isoformat() if timestamps else None,
            "end": max(timestamps).isoformat() if timestamps else None,
        },
        "counts": {
            "detections": len(filtered),
            "uniqueSpecies": len(set(e.get("species") for e in filtered if e.get("species"))),
            "sensorsWithDetections": len(set(e.get("sensorId") for e in filtered if e.get("sensorId"))),
            "microphones": microphones or nodes,
        }
    }


@router.get("/species")
def insights_species(
    limit: int = Query(10, ge=1, le=50),
):
    pipeline = [
        {"$match": {"species": {"$exists": True, "$ne": ""}}},
        {"$group": {"_id": "$species", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
        {"$project": {"_id": 0, "species": "$_id", "count": 1}},
    ]

    return {
        "items": list(db["events"].aggregate(pipeline))
    }
