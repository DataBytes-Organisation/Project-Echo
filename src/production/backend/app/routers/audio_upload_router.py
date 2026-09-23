from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from bson import ObjectId
from datetime import datetime
import os
from typing import Optional
from app.database import AudioUploads

router = APIRouter()

# Ensure the uploads directory exists
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac"}
ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/flac", "audio/x-flac"}
MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # 30 MB


@router.post("/audio/upload")
async def upload_audio(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
):
    # Basic validations
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid audio format. Allowed: .wav, .mp3, .flac")

    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content-type: {file.content_type}")

    # Read to validate size and then write
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 30MB)")

    # Save file with timestamp prefix (sanitize filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    orig_name = os.path.basename(file.filename)
    filename = f"{timestamp}_{orig_name}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(file_path, "wb") as f:
            f.write(data)

        # Store metadata in Mongo
        meta = {
            "original_filename": orig_name,
            "filename": filename,
            "path": os.path.relpath(file_path, os.path.dirname(__file__)),
            "content_type": file.content_type,
            "size_bytes": len(data),
            "upload_timestamp": datetime.utcnow(),
            "user_id": user_id,
        }
        result = AudioUploads.insert_one(meta)
        upload_id = str(result.inserted_id)

        # Enqueue background processing job via RQ
        from app.queue import enqueue_audio_ingest
        job_id = enqueue_audio_ingest(upload_id=upload_id, file_path=file_path, filename=filename)

    except Exception as e:
        # Best-effort cleanup if DB insert fails
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    return {
        "message": "Upload successful",
        "filename": filename,
        "upload_id": upload_id,
        "job_id": job_id,
    }


@router.get("/audio/{upload_id}")
def get_audio(upload_id: str):
    if not ObjectId.is_valid(upload_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid audio upload ID",
        )

    meta = AudioUploads.find_one(
        {"_id": ObjectId(upload_id)}
    )

    if not meta:
        raise HTTPException(
            status_code=404,
            detail="Audio upload not found",
        )

    filename = meta.get("filename")

    if not filename:
        raise HTTPException(
            status_code=404,
            detail="Audio file metadata is incomplete",
        )

    filename = os.path.basename(filename)

    file_path = os.path.join(
        UPLOAD_DIR,
        filename,
    )

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail="Audio file not found on storage",
        )

    return FileResponse(
        path=file_path,
        media_type=meta.get(
            "content_type",
            "application/octet-stream",
        ),
        filename=meta.get(
            "original_filename",
            filename,
        ),
    )
