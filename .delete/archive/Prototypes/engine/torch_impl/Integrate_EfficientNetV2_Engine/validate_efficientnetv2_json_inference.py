"""
JSON-style validation script for EfficientNetV2 inference.

This script simulates how the Engine may receive an MQTT/JSON message.
It validates that the saved EfficientNetV2 predictor can:
1. Read an incoming JSON-style payload.
2. Extract the audio file path.
3. Run preprocessing and inference.
4. Return a structured JSON-style prediction response.

This is not connected to a live MQTT broker yet.
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
# Example JSON/MQTT-style message
# =========================

SAMPLE_MESSAGE = {
    "device_id": "sensor_001",
    "audio_path": r"C:\Deakin\ProjectEcho\Project-Echo\src\Prototypes\data\data_files\Acanthiza chrysorrhoa\region_3.650-4.900.mp3",
    "timestamp": "2026-04-27T15:30:00",
    "source": "dataset_validation"
}


# =========================
# JSON validation functions
# =========================

def validate_input_message(message):
    """
    Validate that the incoming JSON-style message contains the required fields.
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


def run_json_inference(message, model, device, class_mapping, preprocess_config):
    """
    Run EfficientNetV2 inference from a JSON-style input message.
    """

    validate_input_message(message)

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
    print("Starting EfficientNetV2 JSON-style inference validation...")

    print("\nLoading model and configuration files...")
    model, device = load_model(MODEL_PATH)
    class_mapping = load_class_mapping(CLASS_MAPPING_PATH)
    preprocess_config = load_preprocess_config(PREPROCESS_CONFIG_PATH)

    print("Model loaded successfully.")
    print("Device:", device)
    print("Number of classes:", len(class_mapping["classes"]))

    print("\nIncoming JSON-style message:")
    print(json.dumps(SAMPLE_MESSAGE, indent=4))

    print("\nRunning inference...")
    response = run_json_inference(
        message=SAMPLE_MESSAGE,
        model=model,
        device=device,
        class_mapping=class_mapping,
        preprocess_config=preprocess_config
    )

    print("\nInference response:")
    print(json.dumps(response, indent=4))

    output_path = BASE_DIR / "efficientnetv2_json_inference_result.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=4)

    print("\nJSON inference result saved to:")
    print(output_path)


if __name__ == "__main__":
    main()