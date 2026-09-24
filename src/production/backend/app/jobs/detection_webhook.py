"""Outbound detection webhook dispatcher. Runs only after Mongo persistence."""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

import requests
from bson import ObjectId
from bson.errors import InvalidId
from rq import get_current_job

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
    job = get_current_job()
    job_id = job.id if job is not None else "unqueued"
    retries_left = getattr(job, "retries_left", None) if job is not None else None
    attempt_at = datetime.now(timezone.utc).isoformat()
    context = (
        f"job_id={job_id} detection_id={detection_id} "
        f"retries_left={retries_left} attempt_at={attempt_at}"
    )

    webhook_url = (os.getenv("DETECTION_WEBHOOK_URL") or "").strip()
    if not webhook_url:
        logger.info("Detection webhook skipped: DETECTION_WEBHOOK_URL is unset %s", context)
        return False

    try:
        oid = ObjectId(detection_id)
    except (InvalidId, TypeError):
        logger.info("Detection webhook skipped: invalid detection_id %s", context)
        return False

    doc = Detections.find_one({"_id": oid})
    if not doc:
        logger.info("Detection webhook skipped: detection not found %s", context)
        return False

    logger.info("Detection webhook attempt %s", context)
    try:
        response = requests.post(
            webhook_url,
            json=_jsonable_detection(doc),
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.exception(
            "Detection webhook FAILED %s error=%s",
            context,
            exc,
        )
        raise

    logger.info(
        "Detection webhook SUCCEEDED %s status_code=%s final_state=finished",
        context,
        response.status_code,
    )
    return True
