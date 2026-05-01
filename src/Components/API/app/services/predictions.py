import base64
import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException

from app.adapters import AdapterError, AdapterTimeoutError, HttpModelAdapter
from app.database import Predictions


def _raise_prediction_http_error(exc: Exception) -> None:
    if isinstance(exc, AdapterTimeoutError):
        raise HTTPException(status_code=504, detail="Model service timeout")
    if isinstance(exc, AdapterError):
        raise HTTPException(status_code=502, detail=str(exc))
    raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


def _build_inputs_from_audio_bytes(audio_bytes: bytes, filename: str) -> Dict[str, Any]:
    return {
        "audio_b64": base64.b64encode(audio_bytes).decode("utf-8"),
        "filename": filename,
    }


def _build_inputs_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "inputs" in payload:
        return payload["inputs"]
    if "audio_base64" in payload:
        return {"audio_b64": payload["audio_base64"]}
    if "audioClip" in payload:
        return {"audio_b64": payload["audioClip"]}
    raise HTTPException(
        status_code=400,
        detail="Payload must include either 'inputs', 'audio_base64', or 'audioClip'",
    )


def predict_uploaded_audio(
    *,
    filename: str,
    audio_bytes: bytes,
    user_id: Optional[str] = None,
    upload_id: Optional[str] = None,
) -> Dict[str, Any]:
    adapter = HttpModelAdapter()
    model_inputs = _build_inputs_from_audio_bytes(audio_bytes, filename)

    try:
        prediction = adapter.predict(model_inputs)
    except Exception as exc:
        _raise_prediction_http_error(exc)
        raise

    doc = {
        "filename": filename,
        "predicted_species": prediction.species,
        "confidence": prediction.confidence,
        "timestamp": datetime.datetime.utcnow(),
        "user_id": user_id,
    }
    if upload_id:
        doc["upload_id"] = upload_id

    try:
        Predictions.insert_one(doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store prediction: {exc}")

    return {
        "species": prediction.species,
        "confidence": prediction.confidence,
    }


def predict_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    adapter = HttpModelAdapter()
    model_inputs = _build_inputs_from_payload(payload)

    try:
        prediction = adapter.predict(model_inputs)
    except Exception as exc:
        _raise_prediction_http_error(exc)
        raise

    return {
        "species": prediction.species,
        "confidence": prediction.confidence,
        "model_response": prediction.raw,
    }
