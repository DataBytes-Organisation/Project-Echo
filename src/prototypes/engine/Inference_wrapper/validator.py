"""
validator.py

Validation module for the prototype inference wrapper.

Checks:
- Required fields
- Missing input
- Invalid audio
- Prediction validity
"""


class InferenceValidator:

    REQUIRED_FIELDS = [
        "timestamp",
        "sensorId",
        "microphoneLLA",
        "audioClip"
    ]

    @staticmethod
    def validate_required_fields(audio_event):
        """
        Validate that all required fields exist and are not empty.
        """

        missing = []

        for field in InferenceValidator.REQUIRED_FIELDS:
            if field not in audio_event or audio_event[field] in (None, ""):
                missing.append(field)

        if missing:
            return False, {
                "code": "MISSING_REQUIRED_FIELDS",
                "message": f"Missing required fields: {', '.join(missing)}"
            }

        return True, None

    @staticmethod
    def validate_audio(audio_clip):
        """
        Validate the supplied audio clip.
        """

        if audio_clip is None:
            return False, {
                "code": "INVALID_AUDIO",
                "message": "Audio clip is missing."
            }

        if not isinstance(audio_clip, str):
            return False, {
                "code": "INVALID_AUDIO",
                "message": "Audio clip must be Base64 encoded."
            }

        if len(audio_clip.strip()) == 0:
            return False, {
                "code": "INVALID_AUDIO",
                "message": "Audio clip is empty."
            }

        return True, None

    @staticmethod
    def validate_prediction(species, confidence):
        """
        Validate prediction output.
        """

        if species is None:
            return False, {
                "code": "INFERENCE_FAILED",
                "message": "No species predicted."
            }

        if confidence is None:
            return False, {
                "code": "INFERENCE_FAILED",
                "message": "Confidence score missing."
            }

        if confidence < 0 or confidence > 100:
            return False, {
                "code": "INVALID_CONFIDENCE",
                "message": "Confidence must be between 0 and 100."
            }

        return True, None