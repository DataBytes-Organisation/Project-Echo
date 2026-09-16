"""RQ connection for backend jobs (Redis logical DB 1)."""

import logging
import os

from redis import Redis
from rq import Queue, Retry

logger = logging.getLogger(__name__)

QUEUE_NAME = "echo-backend"
DEFAULT_REDIS_URL = "redis://echo-redis:6379/1"
JOB_TIMEOUT_SECONDS = 60
RETRY = Retry(max=3, interval=[10, 30, 60])


def redis_url() -> str:
    return os.getenv("REDIS_URL", DEFAULT_REDIS_URL)


def get_queue() -> Queue:
    return Queue(QUEUE_NAME, connection=Redis.from_url(redis_url()))


def enqueue_detection_webhook(detection_id: str) -> None:
    from app.jobs.detection_webhook import process_detection_job

    try:
        get_queue().enqueue(
            process_detection_job,
            str(detection_id),
            retry=RETRY,
            job_timeout=JOB_TIMEOUT_SECONDS,
        )
        logger.info(
            "Enqueued detection webhook job",
            extra={"job": "detection_webhook", "detection_id": str(detection_id)},
        )
    except Exception:
        logger.exception(
            "Failed to enqueue detection webhook job",
            extra={"job": "detection_webhook", "detection_id": str(detection_id)},
        )
