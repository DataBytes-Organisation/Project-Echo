from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from datetime import datetime
import os
from typing import Optional
from app.database import AudioUploads

router = APIRouter()

# Ensure the uploads directory exists
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


@router.post("/audio/upload")
async def upload_audio(file: UploadFile = File(...), user_id: Optional[str] = Form(None)):
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
    except Exception as e:
        # Best-effort cleanup if DB insert fails
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    return {"message": "Upload successful", "filename": filename, "upload_id": upload_id}
