"""
MQTT-style message handler validation for EfficientNetV2 inference.

This script validates the part of the Engine flow that would handle an MQTT message.
It does not require a live MQTT broker.

It simulates:
1. Receiving an MQTT payload as bytes.
2. Decoding the payload into JSON.
3. Extracting the audio file path.
4. Running EfficientNetV2 preprocessing and inference.
5. Returning a structured prediction response.

Live MQTT broker integration can be added separately after this handler is verified.
"""

import json
import pathlib
from datetime import datetime

from efficientnetv2_predictor import (
    load_model,
    load_class_mapping,
    load_preprocess_config,
    predict_audio
)


# =========================
# Paths
# =========================

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "_trained_models"

MODEL_PATH = MODEL_DIR / "efficientnetv2_project_echo.pt"
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"
PREPROCESS_CONFIG_PATH = MODEL_DIR / "preprocess_config.json"


# =========================
# Example MQTT-style payload
# =========================

SAMPLE_MQTT_MESSAGE = {
    "device_id": "sensor_001",
    "audio_path": r"C:\Deakin\ProjectEcho\Project-Echo\src\Prototypes\data\data_files\Acanthiza chrysorrhoa\region_3.650-4.900.mp3",
    "timestamp": "2026-04-27T15:30:00",
    "source": "mqtt_simulation"
}

# MQTT payload usually arrives as bytes
SAMPLE_MQTT_PAYLOAD = json.dumps(SAMPLE_MQTT_MESSAGE).encode("utf-8")


# =========================
# Validation functions
# =========================

def decode_mqtt_payload(payload):
    """
    Decode an MQTT payload from bytes into a Python dictionary.
    """

    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")

    message = json.loads(payload)
    return message


def validate_mqtt_message(message):
    """
    Validate required fields in the MQTT-style message.
    """

    required_fields = ["device_id", "audio_path", "timestamp"]
    missing_fields = []

    for field in required_fields:
        if field not in message:
            missing_fields.append(field)

    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    audio_path = pathlib.Path(message["audio_path"])

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    return True


def handle_mqtt_message(payload, model, device, class_mapping, preprocess_config):
    """
    Simulate the Engine's MQTT message handler.

    Input:
        payload: MQTT message payload as bytes or string

    Output:
        structured prediction response as dictionary
    """

    message = decode_mqtt_payload(payload)
    validate_mqtt_message(message)

    prediction = predict_audio(
        audio_path=message["audio_path"],
        model=model,
        class_mapping=class_mapping,
        preprocess_config=preprocess_config,
        device=device
    )

    response = {
        "device_id": message["device_id"],
        "input_timestamp": message["timestamp"],
        "processed_timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "EfficientNetV2",
        "input_audio": prediction["audio_path"],
        "prediction": {
            "label": prediction["predicted_label"],
            "class_index": prediction["predicted_index"],
            "confidence": prediction["confidence"]
        },
        "top_predictions": prediction.get("top_predictions", []),
        "status": "success"
    }

    return response


def main():
    print("Starting EfficientNetV2 MQTT message handler validation...")

    print("\nLoading model and configuration files...")
    model, device = load_model(MODEL_PATH)
    class_mapping = load_class_mapping(CLASS_MAPPING_PATH)
    preprocess_config = load_preprocess_config(PREPROCESS_CONFIG_PATH)

    print("Model loaded successfully.")
    print("Device:", device)
    print("Number of classes:", len(class_mapping["classes"]))

    print("\nSimulated MQTT payload:")
    print(SAMPLE_MQTT_PAYLOAD)

    print("\nRunning MQTT-style message handler...")
    response = handle_mqtt_message(
        payload=SAMPLE_MQTT_PAYLOAD,
        model=model,
        device=device,
        class_mapping=class_mapping,
        preprocess_config=preprocess_config
    )

    print("\nMQTT-style inference response:")
    print(json.dumps(response, indent=4))

    output_path = BASE_DIR / "efficientnetv2_mqtt_message_handler_result.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=4)

    print("\nMQTT message handler result saved to:")
    print(output_path)


if __name__ == "__main__":
    main()