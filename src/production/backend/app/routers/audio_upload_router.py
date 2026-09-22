from datetime import datetime
import os
import re
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.database import AudioUploads
from app.services.r2_storage_service import (
    R2StorageUnavailable,
    R2UploadFailed,
    delete_from_r2_best_effort,
    get_r2_storage,
    upload_to_r2,
)


router = APIRouter()

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac"}
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/flac",
    "audio/x-flac",
}
MAX_UPLOAD_BYTES = 30 * 1024 * 1024


def _safe_filename(filename: str) -> str:
    leaf_name = filename.replace("\\", "/").split("/")[-1]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf_name)
    safe_name = safe_name.strip("._")
    return safe_name or "audio"


def _upload_size(file: UploadFile) -> int:
    if file.size is not None:
        return file.size

    current_position = file.file.tell()
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(current_position)
    return size


@router.post("/audio/upload")
async def upload_audio(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
):
    if not file or not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided",
        )

    original_filename = file.filename.replace("\\", "/").split("/")[-1]
    safe_name = _safe_filename(original_filename)

    _, ext = os.path.splitext(safe_name.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio format. Allowed: .wav, .mp3, .flac",
        )

    if (
        file.content_type
        and file.content_type.lower() not in ALLOWED_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content-type: {file.content_type}",
        )

    size_bytes = _upload_size(file)

    if size_bytes == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file",
        )

    if size_bytes > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large (max 30MB)",
        )

    timestamp = datetime.utcnow()
    filename = (
        f"{timestamp:%Y%m%d%H%M%S}_"
        f"{uuid4().hex[:12]}_{safe_name}"
    )
    storage_key = (
        f"audio_uploads/{timestamp:%Y/%m/%d}/{filename}"
    )

    try:
        storage = get_r2_storage()
    except R2StorageUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Audio storage is temporarily unavailable",
        )

    await file.seek(0)

    try:
        await run_in_threadpool(
            upload_to_r2,
            storage,
            file.file,
            storage_key,
            file.content_type,
        )
    except R2StorageUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Audio storage is temporarily unavailable",
        )
    except R2UploadFailed:
        raise HTTPException(
            status_code=502,
            detail="Failed to upload audio to object storage",
        )

    meta = {
        "original_filename": original_filename,
        "filename": filename,
        "storage_key": storage_key,
        "content_type": file.content_type,
        "size_bytes": size_bytes,
        "upload_timestamp": timestamp,
        "user_id": user_id,
    }

    try:
        result = await run_in_threadpool(
            AudioUploads.insert_one,
            meta,
        )
        upload_id = str(result.inserted_id)
    except Exception:
        await run_in_threadpool(
            delete_from_r2_best_effort,
            storage,
            storage_key,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to save upload metadata",
        )

    return {
        "message": "Upload successful",
        "filename": filename,
        "upload_id": upload_id,
        "storage_key": storage_key,
    }
