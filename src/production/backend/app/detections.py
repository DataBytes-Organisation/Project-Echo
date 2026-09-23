import base64
import binascii
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from bson import ObjectId
from fastapi import HTTPException
from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from app.database import Detections, Events
from app.detection_rules import mutable_detection_update, validate_list_filters
from app.exceptions import (
    DetectionNotFoundError,
    DetectionRuleError,
    DetectionStorageError,
)
from app.schemas import DetectionCreate, Detection
from app.detection_rules import evaluate_detection, log_rejected_detection
from app.cache import invalidate_insights


STORAGE_UNAVAILABLE_MESSAGE = "Detection storage is temporarily unavailable."


def _object_id(detection_id: str) -> ObjectId:
    """Parse a detection id or report a client-facing rule violation."""
    if not ObjectId.is_valid(detection_id):
        raise DetectionRuleError("detection_id must be a valid MongoDB ObjectId.")
    return ObjectId(detection_id)


def _raise_storage_error(exc: Exception):
    raise DetectionStorageError(STORAGE_UNAVAILABLE_MESSAGE) from exc

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
    """Build the shared filter used by both offset and cursor pagination."""
    query: Dict[str, Any] = {}

    if species:
        query["species"] = species

    if start_time or end_time:
        timestamp_filter: Dict[str, Any] = {}
        if start_time:
            timestamp_filter["$gte"] = start_time
        if end_time:
            timestamp_filter["$lte"] = end_time
        query["timestamp"] = timestamp_filter

    if lat is not None and lon is not None and radius_km is not None:
        delta_deg = radius_km / 111.0
        query["microphoneLLA.latitude"] = {
            "$gte": lat - delta_deg,
            "$lte": lat + delta_deg,
        }
        query["microphoneLLA.longitude"] = {
            "$gte": lon - delta_deg,
            "$lte": lon + delta_deg,
        }

    return query


def _encode_cursor(timestamp: datetime, detection_id: ObjectId) -> str:
    """Create an opaque keyset cursor from the final item in a page."""
    payload = json.dumps(
        {"timestamp": timestamp.isoformat(), "id": str(detection_id)},
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> Tuple[datetime, ObjectId]:
    """Decode and validate an opaque keyset cursor supplied by a client."""
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        timestamp = datetime.fromisoformat(payload["timestamp"])
        detection_id = ObjectId(payload["id"])
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        binascii.Error,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("Invalid cursor") from exc

    return timestamp, detection_id


def create_detection(detection_in: DetectionCreate) -> Detection:
    accepted, reason = evaluate_detection(detection_in)
    if not accepted:
        log_rejected_detection(detection_in, reason)
        raise HTTPException(status_code=422, detail=f"Detection rejected: {reason}")

    payload = detection_in.dict(by_alias=True)

    try:
        result = Detections.insert_one(payload)
        created = Detections.find_one({"_id": result.inserted_id})
    except PyMongoError as exc:
        _raise_storage_error(exc)

    if not created:
        raise DetectionStorageError(
            "The detection was not available after it was created."
        )

    invalidate_insights()
    return _doc_to_detection(created)


def create_detections_bulk(
    detections_in: List[DetectionCreate],
) -> List[str]:
    """
    Insert multiple validated detections in one MongoDB operation.
    """
    if not detections_in:
        return []

    documents = [
        detection.dict(by_alias=True)
        for detection in detections_in
    ]

    try:
        result = Detections.insert_many(documents)
    except PyMongoError as exc:
        _raise_storage_error(exc)

    invalidate_insights()

    return [
        str(inserted_id)
        for inserted_id in result.inserted_ids
    ]


def get_detection(detection_id: str) -> Detection:
    oid = _object_id(detection_id)
    try:
        doc = Detections.find_one({"_id": oid})
    except PyMongoError as exc:
        _raise_storage_error(exc)

    if not doc:
        raise DetectionNotFoundError("Detection not found.")

    return _doc_to_detection(doc)


def list_real_events(page_size: int = 100) -> Dict[str, Any]:
    """Engine events for the authenticated HMI read; never served elsewhere."""
    query: Dict[str, Any] = {"sourceType": "real"}

    total = Events.count_documents(query)
    cursor = Events.find(query).sort("timestamp", -1).limit(page_size)
    # Raw docs: the /detections-collection Detection contract (list LLAs, no
    # sourceType) must not coerce Engine event reads; the route serializes
    # through eventListEntity and validates against RealDetectionRead.
    items: List[Dict[str, Any]] = list(cursor)

    return {
        "items": items,
        "total": total,
        "page": 1,
        "page_size": page_size,
    }


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
    validate_list_filters(start_time, end_time, lat, lon, radius_km)
    query = _build_detection_query(
        species, start_time, end_time, lat, lon, radius_km
    )

    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20

    skip = (page - 1) * page_size

    try:
        total = Detections.count_documents(query)
        cursor = (
            Detections.find(query)
            .sort("timestamp", -1)
            .skip(skip)
            .limit(page_size)
        )
        items: List[Detection] = [Detection(**doc) for doc in cursor]
    except PyMongoError as exc:
        _raise_storage_error(exc)

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
    """Return detections using a stable timestamp-and-ID keyset cursor."""
    validate_list_filters(start_time, end_time, lat, lon, radius_km)
    query = _build_detection_query(
        species, start_time, end_time, lat, lon, radius_km
    )

    if cursor:
        cursor_timestamp, cursor_id = _decode_cursor(cursor)
        after_cursor = {
            "$or": [
                {"timestamp": {"$lt": cursor_timestamp}},
                {"timestamp": cursor_timestamp, "_id": {"$lt": cursor_id}},
            ]
        }
        query = {"$and": [query, after_cursor]} if query else after_cursor

    try:
        documents = list(
            Detections.find(query)
            .sort([("timestamp", -1), ("_id", -1)])
            .limit(limit + 1)
        )
    except PyMongoError as exc:
        _raise_storage_error(exc)

    next_cursor = None
    if len(documents) > limit:
        last_item = documents[limit - 1]
        next_cursor = _encode_cursor(last_item["timestamp"], last_item["_id"])
        documents = documents[:limit]

    return {
        "items": [_doc_to_detection(document) for document in documents],
        "limit": limit,
        "next_cursor": next_cursor,
    }



def delete_detection(detection_id: str) -> bool:
    oid = _object_id(detection_id)
    try:
        result = Detections.delete_one({"_id": oid})
    except PyMongoError as exc:
        _raise_storage_error(exc)

    if result.deleted_count != 1:
        raise DetectionNotFoundError("Detection not found.")

    invalidate_insights()
    return True


def update_detection(
    detection_id: str,
    update_data: Dict[str, Any],
) -> Detection:
    oid = _object_id(detection_id)

    safe_update = mutable_detection_update(update_data)

    try:
        doc = Detections.find_one_and_update(
            {"_id": oid},
            {"$set": safe_update},
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError as exc:
        _raise_storage_error(exc)

    if not doc:
        raise DetectionNotFoundError("Detection not found.")

    invalidate_insights()
    return _doc_to_detection(doc)
