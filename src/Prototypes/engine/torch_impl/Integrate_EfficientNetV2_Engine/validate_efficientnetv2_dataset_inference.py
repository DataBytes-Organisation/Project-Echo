"""
Dataset validation script for EfficientNetV2 inference.

This script validates the saved EfficientNetV2 model using labelled dataset audio files.
It checks that:
1. The saved model can be loaded.
2. The class mapping can be loaded.
3. The preprocessing configuration can be loaded.
4. Dataset audio files can be preprocessed.
5. The model returns predicted labels and confidence values.
6. Predictions can be compared with the actual dataset folder labels.

This is dataset-based validation only.
"""

import json
import pathlib
import pandas as pd

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

DATA_ROOT = pathlib.Path(
    r"C:\Deakin\ProjectEcho\Project-Echo\src\Prototypes\data\data_files"
)

VALID_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

# Change this number if needed.
# Use 1 for a smoke test.
# Use 20, 50, or 100 for better dataset validation.
MAX_FILES = 50


# =========================
# Dataset scanning
# =========================

def scan_dataset(data_root, valid_exts, max_files=None):
    """
    Scan dataset folder and collect audio files.

    Assumption:
    Each class/species is stored as a folder name.

    Example:
    data_files/
        Acanthiza chrysorrhoa/
            region_3.650-4.900.mp3
    """

    rows = []

    for audio_path in data_root.rglob("*"):
        if audio_path.is_file() and audio_path.suffix.lower() in valid_exts:
            if "_feature_cache" in audio_path.parts:
                continue
            if "_trained_models" in audio_path.parts:
                continue

            actual_label = audio_path.parent.name

            rows.append({
                "audio_path": str(audio_path),
                "actual_label": actual_label
            })

            if max_files is not None and len(rows) >= max_files:
                break

    return rows


# =========================
# Validation
# =========================

def validate_dataset_inference():
    print("Starting EfficientNetV2 dataset validation...")

    print("\nLoading model and configuration files...")

    model, device = load_model(MODEL_PATH)
    class_mapping = load_class_mapping(CLASS_MAPPING_PATH)
    preprocess_config = load_preprocess_config(PREPROCESS_CONFIG_PATH)

    print("Model loaded successfully.")
    print("Device:", device)
    print("Number of classes:", len(class_mapping["classes"]))

    print("\nScanning dataset files...")
    dataset_rows = scan_dataset(
        DATA_ROOT,
        VALID_EXTS,
        max_files=MAX_FILES
    )

    if len(dataset_rows) == 0:
        raise RuntimeError("No audio files found. Please check DATA_ROOT path.")

    print(f"Found {len(dataset_rows)} audio files for validation.")

    results = []

    for row in dataset_rows:
        audio_path = row["audio_path"]
        actual_label = row["actual_label"]

        try:
            prediction = predict_audio(
                audio_path=audio_path,
                model=model,
                class_mapping=class_mapping,
                preprocess_config=preprocess_config,
                device=device
            )

            predicted_label = prediction["predicted_label"]
            confidence = prediction["confidence"]

            is_correct = actual_label == predicted_label

            results.append({
                "audio_path": audio_path,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "correct": is_correct
            })

            print(
                f"Actual: {actual_label} | "
                f"Predicted: {predicted_label} | "
                f"Confidence: {confidence:.4f} | "
                f"Correct: {is_correct}"
            )

        except Exception as e:
            results.append({
                "audio_path": audio_path,
                "actual_label": actual_label,
                "predicted_label": None,
                "confidence": None,
                "correct": False,
                "error": str(e)
            })

            print(f"Error processing {audio_path}: {e}")

    results_df = pd.DataFrame(results)

    total_files = len(results_df)
    correct_count = results_df["correct"].sum()
    accuracy = correct_count / total_files if total_files > 0 else 0

    print("\nValidation summary")
    print("------------------")
    print("Total files tested:", total_files)
    print("Correct predictions:", int(correct_count))
    print("Validation accuracy:", round(accuracy, 4))

    output_path = BASE_DIR / "efficientnetv2_dataset_validation_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nValidation results saved to:")
    print(output_path)

    return results_df


if __name__ == "__main__":
    validate_dataset_inference()