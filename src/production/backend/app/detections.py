from typing import List, Optional, Dict, Any
from datetime import datetime

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from app.database import Detections
from app.detection_rules import mutable_detection_update, validate_list_filters
from app.exceptions import (
    DetectionNotFoundError,
    DetectionRuleError,
    DetectionStorageError,
)
from app.schemas import DetectionCreate, Detection


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


def create_detection(detection_in: DetectionCreate) -> Detection:
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

    return _doc_to_detection(created)


def get_detection(detection_id: str) -> Detection:
    oid = _object_id(detection_id)
    try:
        doc = Detections.find_one({"_id": oid})
    except PyMongoError as exc:
        _raise_storage_error(exc)

    if not doc:
        raise DetectionNotFoundError("Detection not found.")

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
    query: Dict[str, Any] = {}
    validate_list_filters(start_time, end_time, lat, lon, radius_km)

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



def delete_detection(detection_id: str) -> bool:
    oid = _object_id(detection_id)
    try:
        result = Detections.delete_one({"_id": oid})
    except PyMongoError as exc:
        _raise_storage_error(exc)

    if result.deleted_count != 1:
        raise DetectionNotFoundError("Detection not found.")
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

    return _doc_to_detection(doc)
