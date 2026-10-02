"""HTTP API for asynchronous audio processing jobs."""

import asyncio
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from app.database import AudioUploads
from app.routers.audio_upload_router import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    UPLOAD_DIR,
)
from app.services import audio_jobs


router = APIRouter(prefix="/audio/jobs", tags=["audio-jobs"])


def _store_audio_job(
    file_path: str,
    data: bytes,
    filename: str,
    original_filename: str,
    content_type: Optional[str],
    user_id: Optional[str],
):
    """Persist an upload and its job, undoing either partial write on failure."""
    upload_id = None
    try:
        with open(file_path, "wb") as output:
            output.write(data)
        upload = {
            "original_filename": original_filename,
            "filename": filename,
            "path": os.path.relpath(file_path, os.path.dirname(__file__)),
            "content_type": content_type,
            "size_bytes": len(data),
            "upload_timestamp": datetime.utcnow(),
            "user_id": user_id,
        }
        upload_result = AudioUploads.insert_one(upload)
        upload_id = upload_result.inserted_id
        return audio_jobs.create_job(str(upload_id), file_path, filename, user_id)
    except Exception:
        # The upload and processing-job records form one logical operation.
        # Remove any persisted upload metadata as well as the file on failure.
        if upload_id is not None:
            try:
                AudioUploads.delete_one({"_id": upload_id})
            except Exception:
                pass
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass
        raise


@router.post("", status_code=status.HTTP_202_ACCEPTED, summary="Queue an audio file for background prediction")
async def queue_audio_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    _, extension = os.path.splitext(file.filename.lower())
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid audio format. Allowed: .wav, .mp3, .flac")
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content-type: {file.content_type}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 30MB)")

    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{os.path.basename(file.filename)}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    try:
        # File and PyMongo operations are blocking; keep the event loop free for
        # concurrent WebSocket clients while they complete.
        job = await asyncio.to_thread(
            _store_audio_job,
            file_path,
            data,
            filename,
            os.path.basename(file.filename),
            file.content_type,
            user_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to queue audio processing job") from exc

    background_tasks.add_task(audio_jobs.process_job, str(job["_id"]))
    return {"job_id": str(job["_id"]), "status": "queued"}


@router.get("/{job_id}", summary="Get audio processing job status")
def get_audio_job(job_id: str):
    try:
        job = audio_jobs.get_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid job ID") from exc
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return audio_jobs.serialize_job(job)


@router.post("/{job_id}/retry", status_code=status.HTTP_202_ACCEPTED, summary="Retry a failed audio processing job")
def retry_audio_job(job_id: str, background_tasks: BackgroundTasks):
    try:
        audio_jobs._job_id(job_id)
        job = audio_jobs.retry_job(job_id)
    except ValueError as exc:
        status_code = 400 if str(exc) == "Invalid job ID" else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    background_tasks.add_task(audio_jobs.process_job, str(job["_id"]))
    return {"job_id": str(job["_id"]), "status": "queued"}
