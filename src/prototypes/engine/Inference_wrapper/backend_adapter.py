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
        Convert InferenceResponse to the payload currently
        expected by the Backend.
        """

        return {

            "timestamp": response.timestamp,

            "species": response.species,

            "confidence": response.confidence,

            "sensorId": response.sensorId,

            "microphoneLLA": [
                response.microphoneLLA["latitude"],
                response.microphoneLLA["longitude"],
                response.microphoneLLA["altitude"]
            ] if response.microphoneLLA else None,

            "animalEstLLA": [
                response.animalEstLLA["latitude"],
                response.animalEstLLA["longitude"],
                response.animalEstLLA["altitude"]
            ] if response.animalEstLLA else None,

            "animalTrueLLA": [
                response.animalTrueLLA["latitude"],
                response.animalTrueLLA["longitude"],
                response.animalTrueLLA["altitude"]
            ] if response.animalTrueLLA else None,

            "animalLLAUncertainty":
                response.animalLLAUncertainty,

            "audioClip":
                response.audioClip,

            "sampleRate":
                response.sampleRate
        }