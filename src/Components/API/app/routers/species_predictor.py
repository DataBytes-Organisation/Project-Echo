from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

# Placeholder prediction
def predict_species(audio_file: UploadFile):
    return {
        "species": "Crimson Rosella",
        "confidence": 0.92
    }

@router.post("/predict")
async def predict(audio: UploadFile = File(...)):
    if not audio:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    prediction = predict_species(audio)
    return JSONResponse(content=prediction)
