from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
from app.database import Predictions
import datetime
import time
import re
import os

router = APIRouter()

# Model/version metadata and validation config
MODEL_VERSION = "placeholder-v1"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/flac", "audio/x-m4a", "audio/mp4"}
MAX_BYTES = 30 * 1024 * 1024  # 30MB


def _validate_audio(filename: str, content_type: Optional[str], data: bytes):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=422, detail="Unsupported file type; allowed: .wav, .mp3, .m4a, .flac")
    if data is None or len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 30MB)")
    # content_type may be missing from some clients; only enforce if provided
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=422, detail=f"Unsupported Content-Type: {content_type}")


# Placeholder prediction using bytes (replace with real model later)
def predict_species(audio_bytes: bytes):
    # ...existing code...
    return {
        "species": "Crimson Rosella",
        "confidence": 0.92,
        "model_version": MODEL_VERSION,
    }


@router.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    upload_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
):
    # Validate presence
    if not audio or not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Read and validate audio
    try:
        data = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    _validate_audio(audio.filename, getattr(audio, "content_type", None), data)

    # Validate optional upload_id linkage (24-hex)
    valid_upload_id = None
    if upload_id:
        if re.fullmatch(r"[0-9a-fA-F]{24}", upload_id):
            valid_upload_id = upload_id
        else:
            raise HTTPException(status_code=400, detail="Invalid upload_id format")

    # Inference with error handling and timing
    try:
        t0 = time.perf_counter()
        prediction = predict_species(data)
        t1 = time.perf_counter()
        inference_ms = int((t1 - t0) * 1000)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Prediction failed: {e}")

    # Persist prediction result
    doc = {
        "filename": audio.filename,
        "predicted_species": prediction.get("species"),
        "confidence": prediction.get("confidence"),
        "timestamp": datetime.datetime.utcnow(),
        "user_id": user_id,
        "model_version": prediction.get("model_version", MODEL_VERSION),
        "inference_ms": inference_ms,
    }
    if valid_upload_id:
        doc["upload_id"] = valid_upload_id

    try:
        Predictions.insert_one(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store prediction: {e}")

    # Build response
    resp = {
        "species": prediction.get("species"),
        "confidence": prediction.get("confidence"),
        "model_version": doc["model_version"],
        "inference_ms": inference_ms,
    }
    if valid_upload_id:
        resp["upload_id"] = valid_upload_id

    return JSONResponse(content=resp)


@router.get("/predictions/recent")
def recent_predictions(limit: int = 10):
    try:
        docs = list(Predictions.find().sort("_id", -1).limit(int(limit)))
        for d in docs:
            if "_id" in d:
                d["_id"] = str(d["_id"]) 
            ts = d.get("timestamp")
            if isinstance(ts, datetime.datetime):
                d["timestamp"] = ts.replace(tzinfo=None).isoformat() + "Z"
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch predictions: {e}")
