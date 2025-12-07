from typing import List, Optional, Dict, Any

from bson import ObjectId
from pymongo import ReturnDocument

from app.database import Detections
from app.schemas import DetectionCreate, Detection
from typing import Optional

def _doc_to_detection(doc: Dict[str, Any]) -> Optional[Detection]:
    if not doc:
        return None
    return Detection(**doc)


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
    sensor_id: Optional[str] = None,
    species: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[Detection]:
    query: Dict[str, Any] = {}

    if sensor_id is not None:
        query["sensorId"] = sensor_id
    if species is not None:
        query["species"] = species

    cursor = (
        Detections.find(query)
        .skip(skip)
        .limit(limit)
        .sort("timestamp", -1)
    )

    results: List[Detection] = []
    for doc in cursor:
        det = _doc_to_detection(doc)
        if det:
            results.append(det)

    return results


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
