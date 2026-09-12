from unittest.mock import patch, MagicMock

from echo_engine import EchoEngine


def test_echo_engine_sends_backend_payload():
    engine = EchoEngine.__new__(EchoEngine)

    audio_event = {
        "timestamp": "2026-08-30T10:00:00Z",
        "sensorId": "mic_01",
        "microphoneLLA": [-38.143, 144.361, 15],
        "animalEstLLA": [-38.142, 144.360, 15],
        "animalTrueLLA": [-38.142, 144.360, 15],
        "animalLLAUncertainty": 8.5,
        "audioClip": "test-audio"
    }

    mock_response = MagicMock()
    mock_response.text = "OK"

    with patch("echo_engine.requests.post", return_value=mock_response) as mock_post:

        engine.echo_api_send_detection_event(
            audio_event=audio_event,
            sample_rate=48000,
            predicted_class="Koala",
            predicted_probability=0.9642
        )

    mock_post.assert_called_once()

    _, kwargs = mock_post.call_args

    payload = kwargs["json"]

    assert payload["timestamp"] == "2026-08-30T10:00:00Z"
    assert payload["species"] == "Koala"
    assert payload["confidence"] == 0.9642
    assert payload["sensorId"] == "mic_01"

    assert payload["microphoneLLA"] == [
        -38.143,
        144.361,
        15
    ]

    assert payload["animalEstLLA"] == [
        -38.142,
        144.360,
        15
    ]

    assert payload["animalTrueLLA"] == [
        -38.142,
        144.360,
        15
    ]

    assert payload["animalLLAUncertainty"] == 8.5
    assert payload["audioClip"] == "test-audio"
    assert payload["sampleRate"] == 48000

def test_yamnet_model_is_callable_and_extracts_features():
    import numpy as np
    import echo_engine

    engine = echo_engine.EchoEngine.__new__(echo_engine.EchoEngine)

    # 1 second of silent audio at 16 kHz
    wav = np.zeros(16000, dtype=np.float32)

    features = engine.extract_features(
        echo_engine.yamnet_model,
        [wav]
    )

    assert features is not None
    assert features.shape[0] == 1


def test_echo_engine_does_not_send_backend_payload_on_validation_failure():
    import echo_engine

    engine = echo_engine.EchoEngine.__new__(echo_engine.EchoEngine)

    audio_event = {
        "timestamp": "2026-08-30T10:00:00Z",
        "sensorId": "mic_01",
        "microphoneLLA": [-38.143, 144.361, 15],
        "animalEstLLA": [-38.142, 144.360, 15],
        "animalTrueLLA": [-38.142, 144.360, 15],
        "animalLLAUncertainty": 8.5,
        "audioClip": ""
    }

    with patch("echo_engine.requests.post") as mock_post:

        engine.echo_api_send_detection_event(
            audio_event=audio_event,
            sample_rate=48000,
            predicted_class="Koala",
            predicted_probability=0.96
        )

    mock_post.assert_not_called()