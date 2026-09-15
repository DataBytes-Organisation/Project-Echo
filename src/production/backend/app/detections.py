import base64
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from bson import ObjectId
from pymongo import ReturnDocument

from app.database import Detections
from app.schemas import DetectionCreate, Detection


def _doc_to_detection(doc: Dict[str, Any]) -> Optional[Detection]:
    if not doc:
        return None
    return Detection(**doc)


def _build_detection_query(
    species: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
) -> Dict[str, Any]:
    query: Dict[str, Any] = {}

    if species:
        query["species"] = species

    if start_time or end_time:
        ts_filter: Dict[str, Any] = {}
        if start_time:
            ts_filter["$gte"] = start_time
        if end_time:
            ts_filter["$lte"] = end_time
        query["timestamp"] = ts_filter

    if lat is not None and lon is not None and radius_km is not None:
        delta_deg = radius_km / 111.0
        lat_min = lat - delta_deg
        lat_max = lat + delta_deg
        lon_min = lon - delta_deg
        lon_max = lon + delta_deg

        query["microphoneLLA.0"] = {"$gte": lat_min, "$lte": lat_max}
        query["microphoneLLA.1"] = {"$gte": lon_min, "$lte": lon_max}

    return query


def _encode_cursor(timestamp: datetime, detection_id: ObjectId) -> str:
    payload = {"ts": timestamp.isoformat(), "id": str(detection_id)}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def _decode_cursor(cursor: str) -> Tuple[datetime, ObjectId]:
    try:
        decoded = base64.urlsafe_b64decode(cursor.encode()).decode()
        payload = json.loads(decoded)
        cursor_ts = datetime.fromisoformat(payload["ts"])
        cursor_id = ObjectId(payload["id"])
    except (KeyError, ValueError, TypeError):
        raise ValueError("Invalid cursor")

    return cursor_ts, cursor_id


def create_detection(detection_in: DetectionCreate) -> Detection:
    payload = detection_in.dict(by_alias=True)

    result = Detections.insert_one(payload)
    created = Detections.find_one({"_id": result.inserted_id})

    return _doc_to_detection(created)


def get_detection(detection_id: str) -> Optional[Detection]:
    try:
        oid = ObjectId(detection_id)
    except Exception:
        return None

    doc = Detections.find_one({"_id": oid})
    if not doc:
        return None

    return _doc_to_detection(doc)


def list_detections(
    species: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    query = _build_detection_query(
        species=species,
        start_time=start_time,
        end_time=end_time,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
    )

    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20

    skip = (page - 1) * page_size

    total = Detections.count_documents(query)
    cursor = (
        Detections.find(query)
        .sort("timestamp", -1)
        .skip(skip)
        .limit(page_size)
    )

    items: List[Detection] = [Detection(**doc) for doc in cursor]

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def list_detections_cursor(
    species: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    limit: int = 20,
    cursor: Optional[str] = None,
) -> Dict[str, Any]:
    if limit < 1:
        limit = 20

    query = _build_detection_query(
        species=species,
        start_time=start_time,
        end_time=end_time,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
    )

    if cursor:
        cursor_ts, cursor_id = _decode_cursor(cursor)
        cursor_condition = {
            "$or": [
                {"timestamp": {"$lt": cursor_ts}},
                {"timestamp": cursor_ts, "_id": {"$lt": cursor_id}},
            ]
        }
        if query:
            query = {"$and": [query, cursor_condition]}
        else:
            query = cursor_condition

    docs = list(
        Detections.find(query)
        .sort([("timestamp", -1), ("_id", -1)])
        .limit(limit + 1)
    )

    next_cursor = None
    if len(docs) > limit:
        last_seen = docs[limit - 1]
        next_cursor = _encode_cursor(last_seen["timestamp"], last_seen["_id"])
        docs = docs[:limit]

    items: List[Detection] = [Detection(**doc) for doc in docs]

    return {
        "items": items,
        "limit": limit,
        "next_cursor": next_cursor,
    }


def delete_detection(detection_id: str) -> bool:
    try:
        oid = ObjectId(detection_id)
    except Exception:
        return False

    result = Detections.delete_one({"_id": oid})
    return result.deleted_count == 1


def update_detection(
    detection_id: str,
    update_data: Dict[str, Any],
) -> Optional[Detection]:
    try:
        oid = ObjectId(detection_id)
    except Exception:
        return None

    update_data.pop("_id", None)
    update_data.pop("id", None)

    doc = Detections.find_one_and_update(
        {"_id": oid},
        {"$set": update_data},
        return_document=ReturnDocument.AFTER,
    )

    if not doc:
        return None

    return _doc_to_detection(doc)
