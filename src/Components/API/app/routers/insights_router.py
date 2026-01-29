from fastapi import APIRouter, HTTPException
from pymongo import MongoClient

router = APIRouter(tags=["insights"])

# ---------------------------------------------------------
# MongoDB connection
# ---------------------------------------------------------
client = MongoClient("mongodb://ts-mongodb-cont:27017")
db = client["EchoNet"]

events = db["events"]
audio_uploads = db["audio_uploads"]
microphones = db["microphones"]  # or nodes = db["nodes"]


# ---------------------------------------------------------
# 1. TIMESERIES ENDPOINT
# ---------------------------------------------------------
@router.get(
    "/insights/timeseries",
    summary="Get daily or weekly detection counts for charts"
)
def get_timeseries(range: str = "daily"):
    # Validate input
    if range not in ["daily", "weekly"]:
        raise HTTPException(
            status_code=400,
            detail="range must be 'daily' or 'weekly'"
        )

    try:
        if range == "weekly":
            pipeline = [
                {
                    "$group": {
                        "_id": {
                            "year": {"$isoWeekYear": "$timestamp"},
                            "week": {"$isoWeek": "$timestamp"}
                        },
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"_id.year": 1, "_id.week": 1}}
            ]
        else:
            pipeline = [
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": "$timestamp"
                            }
                        },
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"_id": 1}}
            ]

        results = list(events.aggregate(pipeline))

        # Convert _id into a clean "label" field for frontend charts
        formatted = []
        for r in results:
            formatted.append({
                "label": r["_id"],
                "count": r["count"]
            })

        return {"range": range, "data": formatted}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 2. OVERVIEW ENDPOINT
# ---------------------------------------------------------
@router.get(
    "/insights/overview",
    summary="Get high-level summary metrics for Insights dashboard"
)
def get_overview():
    try:
        total_detections = events.count_documents({})
        total_species = len(events.distinct("species"))
        total_sensors = microphones.count_documents({})
        total_audio_uploads = audio_uploads.count_documents({})

        first_event = events.find_one(sort=[("timestamp", 1)])
        last_event = events.find_one(sort=[("timestamp", -1)])

        time_range = {
            "start": first_event["timestamp"] if first_event else None,
            "end": last_event["timestamp"] if last_event else None
        }

        return {
            "total_detections": total_detections,
            "total_species": total_species,
            "total_sensors": total_sensors,
            "total_audio_uploads": total_audio_uploads,
            "time_range": time_range
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 3. TOP SPECIES ENDPOINT
# ---------------------------------------------------------
@router.get(
    "/insights/species",
    summary="Get top detected species with counts"
)
def get_top_species(limit: int = 5):
    try:
        pipeline = [
            {"$group": {"_id": "$species", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]

        results = list(events.aggregate(pipeline))

        formatted = []
        for r in results:
            formatted.append({
                "species": r["_id"],
                "count": r["count"]
            })

        return {"limit": limit, "data": formatted}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 4. LIVE DETECTIONS ENDPOINT (for map)
# ---------------------------------------------------------
@router.get(
    "/insights/detections",
    summary="Get recent detections with location & confidence for live map"
)
def get_detections(limit: int = 100, confidence_min: float = 0.7):
    """
    Returns recent detections with:
    - species, location (latitude/longitude), timestamp, confidence
    - Filters to confidence >= confidence_min (default 70%)
    """
    try:
        pipeline = [
            # Filter by confidence threshold
            {"$match": {"confidence": {"$gte": confidence_min}}},
            # Sort by timestamp descending (most recent first)
            {"$sort": {"timestamp": -1}},
            # Limit results
            {"$limit": limit},
            # Project only needed fields
            {
                "$project": {
                    "_id": 1,
                    "species": 1,
                    "latitude": 1,
                    "longitude": 1,
                    "timestamp": 1,
                    "confidence": 1,
                    "sensorId": 1
                }
            }
        ]

        results = list(events.aggregate(pipeline))

        formatted = []
        for r in results:
            formatted.append({
                "id": str(r.get("_id")),
                "species": r.get("species"),
                "location": {
                    "latitude": r.get("latitude"),
                    "longitude": r.get("longitude")
                },
                "timestamp": r.get("timestamp"),
                "confidence": r.get("confidence"),
                "sensorId": r.get("sensorId")
            })

        return {
            "confidence_threshold": confidence_min,
            "count": len(formatted),
            "data": formatted
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 5. DETECTION DETAILS ENDPOINT
# ---------------------------------------------------------
@router.get(
    "/insights/detections/{detection_id}",
    summary="Get full details for a specific detection"
)
def get_detection_details(detection_id: str):
    """
    Returns complete detection info for a single detection event
    """
    from bson import ObjectId
    
    try:
        detection = events.find_one({"_id": ObjectId(detection_id)})
        
        if not detection:
            raise HTTPException(status_code=404, detail="Detection not found")
        
        return {
            "id": str(detection.get("_id")),
            "species": detection.get("species"),
            "location": {
                "latitude": detection.get("latitude"),
                "longitude": detection.get("longitude")
            },
            "timestamp": detection.get("timestamp"),
            "confidence": detection.get("confidence"),
            "sensorId": detection.get("sensorId"),
            "audioUrl": detection.get("audioUrl"),
            "spectrogram": detection.get("spectrogram")
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))