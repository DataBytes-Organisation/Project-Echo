from fastapi import APIRouter, UploadFile, File, HTTPException
from datetime import datetime
import os

router = APIRouter()

# Ensure the uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/audio/upload")
async def upload_audio(file: UploadFile = File(...)):
    # Validate file format
    if not file.filename.endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(status_code=400, detail="Invalid audio format")

    # Save file with timestamp prefix
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"message": "Upload successful", "filename": filename}
