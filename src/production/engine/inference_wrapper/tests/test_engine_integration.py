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
            predicted_probability=96.42
        )

    mock_post.assert_called_once()

    _, kwargs = mock_post.call_args

    payload = kwargs["json"]

    assert payload["timestamp"] == "2026-08-30T10:00:00Z"
    assert payload["species"] == "Koala"
    assert payload["confidence"] == 96.42
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