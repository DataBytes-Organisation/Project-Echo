from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib

from fastapi.encoders import jsonable_encoder
from bson import ObjectId
from pymongo import ReturnDocument

from app.database import Detections
from app.schemas import DetectionCreate, Detection
from app.cache import (
    DETECTIONS_CACHE_PREFIX,
    get_cached_json,
    set_cached_json,
    invalidate_detection_cache,
)


def _doc_to_detection(doc: Dict[str, Any]) -> Optional[Detection]:
    """
    Convert a MongoDB document into a Detection model.
    """
    if not doc:
        return None

    return Detection(**doc)


def _build_detection_cache_key(
    species: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    page: int = 1,
    page_size: int = 20,
) -> str:
    """
    Generate a unique Redis cache key for each combination
    of filters and pagination parameters.
    """

    raw_key = (
        f"species={species}|"
        f"start_time={start_time}|"
        f"end_time={end_time}|"
        f"lat={lat}|"
        f"lon={lon}|"
        f"radius_km={radius_km}|"
        f"page={page}|"
        f"page_size={page_size}"
    )

    key_hash = hashlib.sha256(
        raw_key.encode("utf-8")
    ).hexdigest()

    return f"{DETECTIONS_CACHE_PREFIX}{key_hash}"


def create_detection(
    detection_in: DetectionCreate,
) -> Detection:
    """
    Create a new detection in MongoDB.

    Any cached detection lists are invalidated because
    the underlying detection data has changed.
    """

    payload = detection_in.dict(by_alias=True)

    result = Detections.insert_one(payload)

    created = Detections.find_one(
        {"_id": result.inserted_id}
    )

    # New detection means previous cached lists may be stale.
    invalidate_detection_cache()

    return _doc_to_detection(created)


def get_detection(
    detection_id: str,
) -> Optional[Detection]:
    """
    Retrieve one detection by MongoDB ObjectId.
    """

    try:
        oid = ObjectId(detection_id)
    except Exception:
        return None

    doc = Detections.find_one(
        {"_id": oid}
    )

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
    """
    Retrieve detections with Redis caching.

    Flow:
        1. Build a cache key from request parameters.
        2. Check Redis.
        3. Return cached response on cache HIT.
        4. Query MongoDB on cache MISS.
        5. Store JSON-safe response in Redis.
        6. Return the JSON-safe response.
    """

    query: Dict[str, Any] = {}

    # Species filter
    if species:
        query["species"] = species

    # Timestamp filters
    if start_time or end_time:
        ts_filter: Dict[str, Any] = {}

        if start_time:
            ts_filter["$gte"] = start_time

        if end_time:
            ts_filter["$lte"] = end_time

        query["timestamp"] = ts_filter

    # Geographic filter
    if (
        lat is not None
        and lon is not None
        and radius_km is not None
    ):
        delta_deg = radius_km / 111.0

        lat_min = lat - delta_deg
        lat_max = lat + delta_deg

        lon_min = lon - delta_deg
        lon_max = lon + delta_deg

        query["microphoneLLA.0"] = {
            "$gte": lat_min,
            "$lte": lat_max,
        }

        query["microphoneLLA.1"] = {
            "$gte": lon_min,
            "$lte": lon_max,
        }

    # Defensive pagination checks
    if page < 1:
        page = 1

    if page_size < 1:
        page_size = 20

    skip = (page - 1) * page_size

    # Each unique request gets its own cache key.
    cache_key = _build_detection_cache_key(
        species=species,
        start_time=start_time,
        end_time=end_time,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        page=page,
        page_size=page_size,
    )

    # -----------------------------
    # REDIS CACHE LOOKUP
    # -----------------------------

    cached_response = get_cached_json(
        cache_key
    )

    if cached_response is not None:
        return cached_response

    # -----------------------------
    # CACHE MISS -> MONGODB
    # -----------------------------

    total = Detections.count_documents(
        query
    )

    cursor = (
        Detections.find(query)
        .sort("timestamp", -1)
        .skip(skip)
        .limit(page_size)
    )

    items: List[Detection] = [
        Detection(**doc)
        for doc in cursor
    ]

    response = {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }

    # Convert Pydantic models, datetime values and MongoDB
    # ObjectIds into values that can safely be encoded as JSON.
    cache_response = jsonable_encoder(
        response,
        custom_encoder={
            ObjectId: str,
        },
    )

    # Save fresh MongoDB result in Redis.
    set_cached_json(
        cache_key,
        cache_response,
    )

    # IMPORTANT:
    # Return the JSON-safe response rather than the original
    # response containing MongoDB ObjectIds.
    return cache_response


def delete_detection(
    detection_id: str,
) -> bool:
    """
    Delete a detection and invalidate cached detection lists.
    """

    try:
        oid = ObjectId(detection_id)
    except Exception:
        return False

    result = Detections.delete_one(
        {"_id": oid}
    )

    if result.deleted_count == 1:
        # Deleted data may still exist in previously cached lists.
        invalidate_detection_cache()
        return True

    return False


def update_detection(
    detection_id: str,
    update_data: Dict[str, Any],
) -> Optional[Detection]:
    """
    Update a detection and invalidate cached detection lists.
    """

    try:
        oid = ObjectId(detection_id)
    except Exception:
        return None

    # Prevent modification of MongoDB identity fields.
    update_data.pop("_id", None)
    update_data.pop("id", None)

    doc = Detections.find_one_and_update(
        {"_id": oid},
        {"$set": update_data},
        return_document=ReturnDocument.AFTER,
    )

    if not doc:
        return None

    # Updated detection means old cached lists may be stale.
    invalidate_detection_cache()

    return _doc_to_detection(doc)