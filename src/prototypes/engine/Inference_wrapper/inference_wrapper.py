"""
inference_wrapper.py

Lightweight Engine-side inference wrapper.

Responsibilities:
- Build a standard success response.
- Build a standard failure response.
- Keep response generation separate from the Engine logic.
"""

from response_schema import InferenceResponse


class InferenceWrapper:
    """
    Creates standard inference responses for the prototype Engine.
    """

    @staticmethod
    def build_success(
        timestamp,
        species,
        confidence,
        sensorId,
        microphoneLLA,
        animalEstLLA,
        animalTrueLLA,
        animalLLAUncertainty,
        audioClip,
        sampleRate,
    ):
        """
        Create a successful inference response.
        """

        return InferenceResponse(
            status="success",
            timestamp=timestamp,
            species=species,
            confidence=confidence,
            sensorId=sensorId,
            microphoneLLA=microphoneLLA,
            animalEstLLA=animalEstLLA,
            animalTrueLLA=animalTrueLLA,
            animalLLAUncertainty=animalLLAUncertainty,
            audioClip=audioClip,
            sampleRate=sampleRate,
            error=None,
        )

    @staticmethod
    def build_failure(
        timestamp,
        sensorId,
        sampleRate,
        error_code,
        error_message,
    ):
        """
        Create a failed inference response.
        """

        return InferenceResponse(
            status="failed",
            timestamp=timestamp,
            species=None,
            confidence=None,
            sensorId=sensorId,
            microphoneLLA=None,
            animalEstLLA=None,
            animalTrueLLA=None,
            animalLLAUncertainty=None,
            audioClip=None,
            sampleRate=sampleRate,
            error={
                "code": error_code,
                "message": error_message,
            },
        )