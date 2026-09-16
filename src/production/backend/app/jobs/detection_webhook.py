"""Outbound detection webhook dispatcher. Runs only after Mongo persistence."""

import logging
import os
from typing import Any, Dict

import requests
from bson import ObjectId
from bson.errors import InvalidId

from app.database import Detections

logger = logging.getLogger(__name__)

HTTP_TIMEOUT_SECONDS = 15


def _jsonable_detection(doc: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(doc)
    if "_id" in payload:
        payload["_id"] = str(payload["_id"])
    for key, value in list(payload.items()):
        if hasattr(value, "isoformat"):
            payload[key] = value.isoformat()
    return payload


def process_detection_job(detection_id: str) -> bool:
    extra = {"job": "detection_webhook", "detection_id": str(detection_id)}
    webhook_url = (os.getenv("DETECTION_WEBHOOK_URL") or "").strip()
    if not webhook_url:
        logger.info("Detection webhook skipped: DETECTION_WEBHOOK_URL is unset", extra=extra)
        return False

    try:
        oid = ObjectId(detection_id)
    except (InvalidId, TypeError):
        logger.info("Detection webhook skipped: invalid detection_id", extra=extra)
        return False

    doc = Detections.find_one({"_id": oid})
    if not doc:
        logger.info("Detection webhook skipped: detection not found", extra=extra)
        return False

    logger.info("Dispatching detection webhook", extra=extra)
    response = requests.post(
        webhook_url,
        json=_jsonable_detection(doc),
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    logger.info("Detection webhook delivered", extra={**extra, "status_code": response.status_code})
    return True
