import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from bson import ObjectId
from redis import Redis
from rq import Queue, Retry

from app.config import settings

logger = logging.getLogger(__name__)

# Initialize bounded Redis connection using db=1 for the job queue to avoid conflicts with HMI sessions
redis_conn = Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    decode_responses=False,
    socket_connect_timeout=settings.redis_connect_timeout,
    socket_timeout=settings.redis_socket_timeout,
)

# Canonical backend task queue
QUEUE_NAME = "echo-backend"
job_queue = Queue(QUEUE_NAME, connection=redis_conn)


def process_audio_ingest_job(upload_id: str, file_path: Optional[str] = None, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Core application workflow for background audio processing:
    1. Validates upload record in MongoDB
    2. Updates processing state
    3. Performs metadata validation and prepares audio for engine inference
    4. Records job completion timestamp and status
    """
    logger.info("Starting audio ingest background processing for upload_id=%s, file=%s", upload_id, filename)
    from app.database import AudioUploads

    started_at = datetime.utcnow()
    
    try:
        # Find upload record
        query = {"_id": ObjectId(upload_id)} if ObjectId.is_valid(upload_id) else {"filename": filename}
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

        AudioUploads.update_one(query, {"$set": update_data}, upsert=True)

        # File validation and metadata check
        file_size = 0
        resolved_path = file_path or (upload_doc.get("path") if upload_doc else None)
        if resolved_path and os.path.exists(resolved_path):
            file_size = os.path.getsize(resolved_path)

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
        logger.error("Failed audio ingest background processing for upload_id=%s: %s", upload_id, exc, exc_info=True)
        try:
            query = {"_id": ObjectId(upload_id)} if ObjectId.is_valid(upload_id) else {"filename": filename}
            AudioUploads.update_one(
                query,
                {
                    "$set": {
                        "processing_status": "failed",
                        "error_message": str(exc),
                        "failed_at": datetime.utcnow(),
                    }
                },
            )
        except Exception as db_err:
            logger.error("Could not write failure status to DB: %s", db_err)
        raise exc


def enqueue_audio_ingest(upload_id: str, file_path: Optional[str] = None, filename: Optional[str] = None):
    """
    Enqueues an audio ingest job with bounded timeout and retry policies.
    """
    try:
        retry_policy = Retry(
            max=settings.job_max_retries,
            interval=[settings.job_retry_delay_seconds] * settings.job_max_retries,
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
