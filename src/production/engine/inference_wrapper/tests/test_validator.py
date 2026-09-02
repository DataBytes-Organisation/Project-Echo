from ..validator import InferenceValidator


def test_valid_required_fields():
    audio_event = {
        "timestamp": "2026-08-30T10:00:00Z",
        "sensorId": "mic_01",
        "microphoneLLA": [-38.143, 144.361, 15],
        "audioClip": "VGhpcyBpcyBhIHRlc3Q="
    }

    valid, error = InferenceValidator.validate_required_fields(
        audio_event
    )

    assert valid is True
    assert error is None


def test_missing_sensor_id():
    audio_event = {
        "timestamp": "2026-08-30T10:00:00Z",
        "microphoneLLA": [-38.143, 144.361, 15],
        "audioClip": "VGhpcyBpcyBhIHRlc3Q="
    }

    valid, error = InferenceValidator.validate_required_fields(
        audio_event
    )

    assert valid is False
    assert error["code"] == "MISSING_REQUIRED_FIELDS"


def test_invalid_audio():
    valid, error = InferenceValidator.validate_audio("")

    assert valid is False
    assert error["code"] == "INVALID_AUDIO"


def test_invalid_confidence():
    valid, error = InferenceValidator.validate_prediction(
        "Koala",
        120
    )

    assert valid is False
    assert error["code"] == "INVALID_CONFIDENCE"


def test_invalid_sample_rate():
    valid, error = InferenceValidator.validate_sample_rate(0)

    assert valid is False
    assert error["code"] == "INVALID_SAMPLE_RATE"


def test_valid_sample_rate():
    valid, error = InferenceValidator.validate_sample_rate(48000)

    assert valid is True
    assert error is None