"""
backend_adapter.py

Converts the proposed inference response into the
current Backend detection payload.

This allows testing without changing the production Backend.
"""


class BackendAdapter:

    @staticmethod
    def to_backend_payload(response):
        """
        Convert InferenceResponse to the payload expected by the Backend.

        LLA fields are passed through as {latitude, longitude, altitude}
        objects. This adapter originally flattened them back into
        [lat, lon, alt] arrays to match the older Backend contract, but the
        agreed integration schema now uses objects end to end, so flattening
        here silently reverted that contract.
        """

        return {

            "timestamp": response.timestamp,

            "species": response.species,

            "confidence": response.confidence,

            "sensorId": response.sensorId,

            "microphoneLLA": response.microphoneLLA,

            "animalEstLLA": response.animalEstLLA,

            "animalTrueLLA": response.animalTrueLLA,

            "animalLLAUncertainty":
                response.animalLLAUncertainty,

            "audioClip":
                response.audioClip,

            "sampleRate":
                response.sampleRate
        }