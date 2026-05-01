from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
import datetime
import re
from app.database import Predictions
from app.services.predictions import predict_uploaded_audio

router = APIRouter()

@router.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    upload_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
):
    if not audio or not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # If provided, validate upload_id format (24-hex) and store it for linkage
    valid_upload_id = None
    if upload_id:
        if re.fullmatch(r"[0-9a-fA-F]{24}", upload_id):
            valid_upload_id = upload_id
        else:
            raise HTTPException(status_code=400, detail="Invalid upload_id format")

    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file provided")
        prediction = predict_uploaded_audio(
            filename=audio.filename,
            audio_bytes=audio_bytes,
            user_id=user_id,
            upload_id=valid_upload_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return JSONResponse(content=prediction)


@router.get("/predictions/recent")
def recent_predictions(limit: int = 10):
    try:
        docs = list(Predictions.find().sort("_id", -1).limit(int(limit)))
        for d in docs:
            if "_id" in d:
                d["_id"] = str(d["_id"])  # jsonify
            if isinstance(d.get("timestamp"), datetime.datetime):
                d["timestamp"] = d["timestamp"].isoformat()
        return {"count": len(docs), "items": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch predictions: {e}")
