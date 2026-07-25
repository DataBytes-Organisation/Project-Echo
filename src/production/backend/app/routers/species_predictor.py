from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
from app.database import Predictions
from app.services.model_adapter import MultiModalPredictionError, predict_with_failure_detection
import datetime
import re

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
        prediction = predict_with_failure_detection(audio)
    except MultiModalPredictionError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Persist prediction result
    try:
        doc = {
            "filename": audio.filename,
            "predicted_species": prediction["species"],
            "confidence": prediction["confidence"],
            "timestamp": datetime.datetime.utcnow(),
            "user_id": user_id,
        }
        if valid_upload_id:
            doc["upload_id"] = valid_upload_id

        Predictions.insert_one(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store prediction: {e}")

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
