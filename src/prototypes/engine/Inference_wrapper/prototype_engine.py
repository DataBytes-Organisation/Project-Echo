"""
prototype_engine.py


prototype Engine generates the
standard inference response without modifying the
production Engine.
"""

from inference_wrapper import InferenceWrapper
from validator import InferenceValidator
from backend_adapter import BackendAdapter


class PrototypeEngine:

    @staticmethod
    def process_prediction(audio_event,
                           predicted_class,
                           predicted_probability,
                           sample_rate):
        """
        Simulate the prototype Engine inference flow.
        """

        # -----------------------------
        # Validate required fields
        # -----------------------------

        valid, error = InferenceValidator.validate_required_fields(audio_event)

        if not valid:
            return InferenceWrapper.build_failure(
                timestamp=audio_event.get("timestamp", ""),
                sensorId=audio_event.get("sensorId", ""),
                sampleRate=sample_rate,
                error_code=error["code"],
                error_message=error["message"],
            )

        # -----------------------------
        # Validate audio
        # -----------------------------

        valid, error = InferenceValidator.validate_audio(
            audio_event["audioClip"]
        )

        if not valid:
            return InferenceWrapper.build_failure(
                timestamp=audio_event["timestamp"],
                sensorId=audio_event["sensorId"],
                sampleRate=sample_rate,
                error_code=error["code"],
                error_message=error["message"],
            )

        # -----------------------------
        # Validate prediction
        # -----------------------------

        valid, error = InferenceValidator.validate_prediction(
            predicted_class,
            predicted_probability,
        )

        if not valid:
            return InferenceWrapper.build_failure(
                timestamp=audio_event["timestamp"],
                sensorId=audio_event["sensorId"],
                sampleRate=sample_rate,
                error_code=error["code"],
                error_message=error["message"],
            )

        # -----------------------------
        # Create standard response
        # -----------------------------

        response = InferenceWrapper.build_success(

            timestamp=audio_event["timestamp"],

            species=predicted_class,

            confidence=predicted_probability,

            sensorId=audio_event["sensorId"],

            microphoneLLA={
                "latitude": audio_event["microphoneLLA"][0],
                "longitude": audio_event["microphoneLLA"][1],
                "altitude": audio_event["microphoneLLA"][2],
            },

            animalEstLLA={
                "latitude": audio_event["animalEstLLA"][0],
                "longitude": audio_event["animalEstLLA"][1],
                "altitude": audio_event["animalEstLLA"][2],
            },

            animalTrueLLA={
                "latitude": audio_event["animalTrueLLA"][0],
                "longitude": audio_event["animalTrueLLA"][1],
                "altitude": audio_event["animalTrueLLA"][2],
            },

            animalLLAUncertainty=audio_event["animalLLAUncertainty"],

            audioClip=audio_event["audioClip"],

            sampleRate=sample_rate,
        )

        return response

    @staticmethod
    def create_backend_payload(response):
        """
        Convert the standard response into the payload
        expected by the current Backend.
        """

        return BackendAdapter.to_backend_payload(response)