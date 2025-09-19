# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from .overlap_detector import run_from_bytes

app = FastAPI(title="EchoNet Overlap Detector", version="0.2.0")

class OverlapEvent(BaseModel):
    labels: list[str]
    start: float
    end: float
    confidence: float

class PredictResponse(BaseModel):
    events: list[OverlapEvent]
    meta: dict

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be audio/*")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    events, meta = run_from_bytes(data)
    return {"events": events, "meta": meta}
