from numbers import Number
from typing import Any, Dict


class MultiModalPredictionError(Exception):
    """Raised when the multi-modal prediction path fails or returns invalid data."""


def predict_multimodal(audio_file: Any) -> Dict[str, Any]:
    # Placeholder until the real multi-modal engine is wired into the adapter.
    return {
        "species": "Crimson Rosella",
        "confidence": 0.92,
    }


def validate_prediction_response(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise MultiModalPredictionError("Multi-modal model returned an invalid response type")

    species = result.get("species")
    confidence = result.get("confidence")

    if not isinstance(species, str) or not species.strip():
        raise MultiModalPredictionError("Multi-modal model response is missing a valid species")

    if not isinstance(confidence, Number):
        raise MultiModalPredictionError("Multi-modal model response is missing a valid confidence")

    return {
        "species": species,
        "confidence": float(confidence),
    }


def predict_with_failure_detection(audio_file: Any) -> Dict[str, Any]:
    try:
        result = predict_multimodal(audio_file)
    except MultiModalPredictionError:
        raise
    except Exception as exc:
        raise MultiModalPredictionError("Multi-modal prediction failed") from exc

    return validate_prediction_response(result)
