import os
from fastapi import APIRouter, Query
from datetime import datetime
from typing import Optional
import pymongo

router = APIRouter(prefix="/insights", tags=["insights"])

MONGO_URI = os.getenv("MONGO_URI", "mongodb://db:27017")
MONGO_DB = os.getenv("MONGO_DB", "project_echo")

_client = pymongo.MongoClient(MONGO_URI)
db = _client[MONGO_DB]


def parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@router.get("/overview")
def get_insights_overview(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    events = db["events"]
    microphones = db["microphones"]

    start_dt = parse_iso(start)
    end_dt = parse_iso(end)

    match = {}
    if start_dt or end_dt:
        match["timestamp"] = {}
        if start_dt:
            match["timestamp"]["$gte"] = start_dt
        if end_dt:
            match["timestamp"]["$lte"] = end_dt

    pipeline = []
    if match:
        pipeline.append({"$match": match})

    pipeline.extend([
        {
            "$group": {
                "_id": None,
                "detections": {"$sum": 1},
                "speciesSet": {"$addToSet": "$species"},
                "sensorSet": {"$addToSet": "$sensorId"},
                "minTs": {"$min": "$timestamp"},
                "maxTs": {"$max": "$timestamp"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "detections": 1,
                "uniqueSpecies": {"$size": "$speciesSet"},
                "sensorsWithDetections": {"$size": "$sensorSet"},
                "minTs": 1,
                "maxTs": 1,
            }
        }
    ])

    result = list(events.aggregate(pipeline))
    stats = result[0] if result else {
        "detections": 0,
        "uniqueSpecies": 0,
        "sensorsWithDetections": 0,
        "minTs": None,
        "maxTs": None,
    }

    return {
        "timeRange": {
            "start": stats["minTs"].isoformat() if stats["minTs"] else None,
            "end": stats["maxTs"].isoformat() if stats["maxTs"] else None,
        },
        "counts": {
            "detections": stats["detections"],
            "uniqueSpecies": stats["uniqueSpecies"],
            "sensorsWithDetections": stats["sensorsWithDetections"],
            "microphones": microphones.count_documents({}),
        }
    }


@router.get("/species")
def get_insights_species(
    limit: int = Query(10, ge=1, le=100),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    events = db["events"]

    start_dt = parse_iso(start)
    end_dt = parse_iso(end)

    match = {"species": {"$nin": [None, ""]}}
    if start_dt or end_dt:
        match["timestamp"] = {}
        if start_dt:
            match["timestamp"]["$gte"] = start_dt
        if end_dt:
            match["timestamp"]["$lte"] = end_dt

    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$species", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
        {"$project": {"_id": 0, "species": "$_id", "count": 1}},
    ]

    return {"items": list(events.aggregate(pipeline))}
