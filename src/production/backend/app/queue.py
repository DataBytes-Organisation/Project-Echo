import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from bson import ObjectId
from redis import Redis
from rq import Queue, Retry, get_current_job

from app.config import settings

logger = logging.getLogger(__name__)

# Bounded Redis connection on the job-queue DB (REDIS_URL, DB 1) agreed with the
# C1.2 worker (#1048), so jobs never collide with HMI sessions on DB 0.
redis_conn = Redis.from_url(
    settings.redis_url,
    decode_responses=False,
    socket_connect_timeout=settings.redis_connect_timeout,
    socket_timeout=settings.redis_socket_timeout,
)

# Canonical backend task queue
QUEUE_NAME = "echo-backend"
job_queue = Queue(QUEUE_NAME, connection=redis_conn)

# Upload records store `path` relative to the upload router's directory.
_UPLOAD_PATH_BASE = os.path.join(os.path.dirname(__file__), "routers")


class AudioFileMissingError(FileNotFoundError):
    """The uploaded audio is not on storage; retrying cannot fix this."""


def _resolve_audio_path(file_path: Optional[str], upload_doc: Optional[Dict[str, Any]]) -> Optional[str]:
    path = file_path or (upload_doc or {}).get("path")
    if path and not os.path.isabs(path):
        path = os.path.join(_UPLOAD_PATH_BASE, path)
    return path


def _stop_retries() -> None:
    # RQ checks retries_left on this same Job object when handling the failure.
    job = get_current_job()
    if job is not None:
        job.retries_left = 0


def _will_retry() -> bool:
    job = get_current_job()
    return bool(job is not None and job.retries_left)


def process_audio_ingest_job(upload_id: str, file_path: Optional[str] = None, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Core application workflow for background audio processing:
    1. Validates upload record in MongoDB
    2. Updates processing state
    3. Verifies the audio file exists on storage and prepares it for engine inference
    4. Records job completion timestamp and status

    A missing or empty audio file fails the job permanently (no retries); any
    other error is re-raised so RQ retries it on the configured schedule.
    """
    logger.info("Starting audio ingest background processing for upload_id=%s, file=%s", upload_id, filename)
    from app.database import AudioUploads

    started_at = datetime.utcnow()
    query = {"_id": ObjectId(upload_id)} if ObjectId.is_valid(upload_id) else {"filename": filename}

    try:
        upload_doc = AudioUploads.find_one(query)

        if not upload_doc:
            logger.warning("Upload record %s not found in database; initializing stub record.", upload_id)
            update_data = {
                "upload_id": upload_id,
                "filename": filename,
                "processing_status": "processing",
                "processing_started_at": started_at,
            }
        else:
            update_data = {
                "processing_status": "processing",
                "processing_started_at": started_at,
            }

        AudioUploads.update_one(
            query,
            {
                "$set": update_data,
                "$inc": {"processing_attempts": 1},
                "$unset": {"pipeline_status": "", "error_message": ""},
            },
            upsert=True,
        )

        # File validation: never hand a missing file to inference.
        resolved_path = _resolve_audio_path(file_path, upload_doc)
        if not resolved_path or not os.path.isfile(resolved_path):
            raise AudioFileMissingError(f"Audio file not found on storage: {resolved_path or '<no path recorded>'}")
        file_size = os.path.getsize(resolved_path)
        if file_size == 0:
            raise AudioFileMissingError(f"Audio file is empty on storage: {resolved_path}")

        completed_at = datetime.utcnow()
        elapsed_seconds = (completed_at - started_at).total_seconds()

        completion_data = {
            "processing_status": "completed",
            "processing_completed_at": completed_at,
            "processing_duration_seconds": elapsed_seconds,
            "pipeline_status": "ready_for_inference",
            "file_size_verified": file_size,
        }

        AudioUploads.update_one(query, {"$set": completion_data})
        logger.info("Successfully completed audio ingest processing for upload_id=%s in %.2fs", upload_id, elapsed_seconds)

        return {
            "status": "completed",
            "upload_id": upload_id,
            "filename": filename,
            "duration_seconds": elapsed_seconds,
            "completed_at": completed_at.isoformat(),
        }

    except Exception as exc:
        if isinstance(exc, AudioFileMissingError):
            _stop_retries()
        retrying = _will_retry()
        logger.error(
            "Failed audio ingest background processing for upload_id=%s (%s): %s",
            upload_id,
            "retry scheduled" if retrying else "giving up",
            exc,
            exc_info=True,
        )
        try:
            AudioUploads.update_one(
                query,
                {
                    "$set": {
                        "processing_status": "retry_scheduled" if retrying else "failed",
                        "error_message": str(exc),
                        "failed_at": datetime.utcnow(),
                    }
                },
            )
        except Exception as db_err:
            logger.error("Could not write failure status to DB: %s", db_err)
        raise


def enqueue_audio_ingest(upload_id: str, file_path: Optional[str] = None, filename: Optional[str] = None):
    """
    Enqueues an audio ingest job with bounded timeout and retry policies.
    """
    try:
        retry_policy = Retry(
            max=len(settings.job_retry_intervals),
            interval=settings.job_retry_intervals,
        )
        job = job_queue.enqueue(
            process_audio_ingest_job,
            upload_id=upload_id,
            file_path=file_path,
            filename=filename,
            job_timeout=settings.job_timeout_seconds,
            retry=retry_policy,
        )
        logger.info("Enqueued audio ingest job %s for upload %s", job.id, upload_id)
        return job.id
    except Exception as e:
        logger.warning("Failed to enqueue job to Redis queue: %s (continuing synchronously)", e)
        return None
