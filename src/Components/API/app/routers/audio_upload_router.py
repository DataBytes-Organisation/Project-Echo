from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from app.database import AudioUploads
import os
import datetime

router = APIRouter()

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/flac", "audio/x-m4a", "audio/mp4"}
MAX_BYTES = 30 * 1024 * 1024  # 30MB
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)


def _validate_upload(filename: str, content_type: Optional[str], size: int):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=422, detail="Unsupported file type; allowed: .wav, .mp3, .m4a, .flac")
    if size <= 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if size > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 30MB)")
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=422, detail=f"Unsupported Content-Type: {content_type}")


@router.post("/api/audio/upload")
async def upload_audio(file: UploadFile = File(...), user_id: Optional[str] = Form(None)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Read content to enforce size/empty checks
    try:
        data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    _validate_upload(file.filename, getattr(file, "content_type", None), len(data))

    # Save file with timestamped name
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    safe_name = f"{timestamp}_{os.path.basename(file.filename)}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    try:
        with open(save_path, "wb") as f:
            f.write(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Persist metadata
    doc = {
        "filename": safe_name,
        "path": save_path,
        "content_type": getattr(file, "content_type", None),
        "size": len(data),
        "upload_timestamp": datetime.datetime.utcnow(),
        "user_id": user_id,
    }
    try:
        result = AudioUploads.insert_one(doc)
        upload_id = str(result.inserted_id)
    except Exception as e:
        # Cleanup file on DB failure
        try:
            os.remove(save_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to store metadata: {e}")

    return {"message": "Upload successful", "filename": safe_name, "upload_id": upload_id}
