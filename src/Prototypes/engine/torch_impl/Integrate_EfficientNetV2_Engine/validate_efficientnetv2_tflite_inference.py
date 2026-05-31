"""
TFLite validation script for EfficientNetV2 inference.

This script validates the converted EfficientNetV2 TFLite model using the same
audio preprocessing and class mapping used by the PyTorch predictor.

It performs:
1. Single audio validation
2. PyTorch vs TFLite output comparison
3. Dataset-based TFLite validation
4. CSV result export

This confirms whether the converted TFLite model can be used as a production-ready
model candidate after PyTorch -> ONNX -> TensorFlow SavedModel -> TFLite conversion.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from efficientnetv2_predictor import (
    load_model,
    load_class_mapping,
    load_preprocess_config,
    preprocess_audio,
    predict_audio,
)


# =========================
# Paths
# =========================

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "_trained_models"

PYTORCH_MODEL_PATH = MODEL_DIR / "efficientnetv2_project_echo.pt"
TFLITE_MODEL_PATH = MODEL_DIR / "efficientnetv2_project_echo.tflite"
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"
PREPROCESS_CONFIG_PATH = MODEL_DIR / "preprocess_config.json"

DATA_ROOT = pathlib.Path(
    r"C:\Deakin\ProjectEcho\Project-Echo\src\Prototypes\data\data_files"
)

VALID_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

# Change this if you want to validate more files.
MAX_FILES = 50

# Use one known sample first.
SAMPLE_AUDIO_PATH = pathlib.Path(
    r"C:\Deakin\ProjectEcho\Project-Echo\src\Prototypes\data\data_files\Acanthiza chrysorrhoa\region_3.650-4.900.mp3"
)


# =========================
# TFLite helper functions
# =========================

def load_tflite_interpreter(tflite_model_path):
    """
    Load the TFLite model interpreter.
    """

    if not pathlib.Path(tflite_model_path).exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_model_path}")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def adapt_to_tflite_input(x_torch, input_shape, input_dtype):
    """
    Convert PyTorch-style input tensor to the shape expected by the TFLite model.

    PyTorch preprocessing returns shape:
        [1, 1, 128, 313]  -> NCHW

    TFLite may expect:
        [1, 1, 128, 313]  -> NCHW
    or:
        [1, 128, 313, 1]  -> NHWC
    """

    x = x_torch.detach().cpu().numpy().astype(np.float32)

    if tuple(x.shape) == tuple(input_shape):
        return x.astype(input_dtype)

    x_nhwc = np.transpose(x, (0, 2, 3, 1))

    if tuple(x_nhwc.shape) == tuple(input_shape):
        return x_nhwc.astype(input_dtype)

    raise ValueError(
        f"Could not adapt input shape. "
        f"TFLite expects {input_shape}, but prepared NCHW={x.shape}, NHWC={x_nhwc.shape}"
    )


def run_tflite_prediction(audio_path, interpreter, input_details, output_details, class_mapping, preprocess_config):
    """
    Run one TFLite prediction using the same preprocessing config.
    """

    input_index = input_details[0]["index"]
    input_shape = tuple(input_details[0]["shape"])
    input_dtype = input_details[0]["dtype"]

    x_torch = preprocess_audio(audio_path, preprocess_config)
    x_tflite = adapt_to_tflite_input(x_torch, input_shape, input_dtype)

    interpreter.set_tensor(input_index, x_tflite)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])

    logits_or_probs = output[0].astype(np.float32)

    # Apply softmax because exported model usually returns logits.
    exp_values = np.exp(logits_or_probs - np.max(logits_or_probs))
    probs = exp_values / np.sum(exp_values)

    pred_index = int(np.argmax(probs))
    confidence = float(probs[pred_index])

    index_to_label = class_mapping["index_to_label"]
    pred_label = index_to_label[str(pred_index)]

    top_indices = np.argsort(probs)[::-1][:5]

    top_predictions = [
        {
            "index": int(i),
            "label": str(index_to_label[str(int(i))]),
            "confidence": float(probs[i])
        }
        for i in top_indices
    ]

    return {
        "audio_path": str(audio_path),
        "predicted_index": int(pred_index),
        "predicted_label": str(pred_label),
        "confidence": float(confidence),
        "top_predictions": top_predictions,
        "raw_output_shape": [int(v) for v in output.shape],
        "tflite_input_shape": [int(v) for v in input_shape],
    }


# =========================
# Dataset scanning
# =========================

def scan_dataset(data_root, valid_exts, max_files=None):
    """
    Scan dataset folder and collect audio files.

    Assumption:
    folder name = actual species label
    """

    rows = []

    for audio_path in data_root.rglob("*"):
        if audio_path.is_file() and audio_path.suffix.lower() in valid_exts:
            if "_feature_cache" in audio_path.parts:
                continue
            if "_trained_models" in audio_path.parts:
                continue

            rows.append({
                "audio_path": str(audio_path),
                "actual_label": audio_path.parent.name
            })

            if max_files is not None and len(rows) >= max_files:
                break

    return rows


# =========================
# Validation functions
# =========================

def single_audio_comparison():
    """
    Compare PyTorch and TFLite prediction on one sample audio file.
    """

    print("\n" + "=" * 70)
    print("Single audio PyTorch vs TFLite validation")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pytorch_model, device = load_model(PYTORCH_MODEL_PATH, device=device)
    class_mapping = load_class_mapping(CLASS_MAPPING_PATH)
    preprocess_config = load_preprocess_config(PREPROCESS_CONFIG_PATH)

    interpreter, input_details, output_details = load_tflite_interpreter(TFLITE_MODEL_PATH)

    print("PyTorch model loaded.")
    print("TFLite model loaded.")
    print("Device:", device)
    print("Number of classes:", len(class_mapping["classes"]))

    print("\nTFLite input details:")
    print(input_details[0])

    print("\nTFLite output details:")
    print(output_details[0])

    pytorch_result = predict_audio(
        audio_path=SAMPLE_AUDIO_PATH,
        model=pytorch_model,
        class_mapping=class_mapping,
        preprocess_config=preprocess_config,
        device=device
    )

    tflite_result = run_tflite_prediction(
        audio_path=SAMPLE_AUDIO_PATH,
        interpreter=interpreter,
        input_details=input_details,
        output_details=output_details,
        class_mapping=class_mapping,
        preprocess_config=preprocess_config
    )

    print("\nSample audio:")
    print(SAMPLE_AUDIO_PATH)

    print("\nPyTorch prediction:")
    print(json.dumps(pytorch_result, indent=4))

    print("\nTFLite prediction:")
    print(json.dumps(tflite_result, indent=4))

    label_match = pytorch_result["predicted_label"] == tflite_result["predicted_label"]

    print("\nComparison summary:")
    print("PyTorch label:", pytorch_result["predicted_label"])
    print("TFLite label:", tflite_result["predicted_label"])
    print("Labels match:", label_match)
    print("PyTorch confidence:", pytorch_result["confidence"])
    print("TFLite confidence:", tflite_result["confidence"])

    output_path = BASE_DIR / "efficientnetv2_tflite_single_audio_comparison.json"

    comparison = {
        "sample_audio": str(SAMPLE_AUDIO_PATH),
        "pytorch_result": pytorch_result,
        "tflite_result": tflite_result,
        "labels_match": label_match
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)

    print("\nSingle audio comparison saved to:")
    print(output_path)


def dataset_tflite_validation():
    """
    Validate the TFLite model using multiple labelled dataset audio files.
    """

    print("\n" + "=" * 70)
    print("Dataset-based TFLite validation")
    print("=" * 70)

    class_mapping = load_class_mapping(CLASS_MAPPING_PATH)
    preprocess_config = load_preprocess_config(PREPROCESS_CONFIG_PATH)

    interpreter, input_details, output_details = load_tflite_interpreter(TFLITE_MODEL_PATH)

    dataset_rows = scan_dataset(DATA_ROOT, VALID_EXTS, max_files=MAX_FILES)

    if len(dataset_rows) == 0:
        raise RuntimeError("No audio files found. Please check DATA_ROOT.")

    print(f"Found {len(dataset_rows)} files for TFLite validation.")

    results = []

    for row in dataset_rows:
        audio_path = row["audio_path"]
        actual_label = row["actual_label"]

        try:
            result = run_tflite_prediction(
                audio_path=audio_path,
                interpreter=interpreter,
                input_details=input_details,
                output_details=output_details,
                class_mapping=class_mapping,
                preprocess_config=preprocess_config
            )

            predicted_label = result["predicted_label"]
            confidence = result["confidence"]
            correct = actual_label == predicted_label

            results.append({
                "audio_path": audio_path,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "correct": correct
            })

            print(
                f"Actual: {actual_label} | "
                f"TFLite Predicted: {predicted_label} | "
                f"Confidence: {confidence:.4f} | "
                f"Correct: {correct}"
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
    correct_count = int(results_df["correct"].sum())
    accuracy = correct_count / total_files if total_files > 0 else 0

    print("\nTFLite validation summary")
    print("-------------------------")
    print("Total files tested:", total_files)
    print("Correct predictions:", correct_count)
    print("TFLite validation accuracy:", round(accuracy, 4))

    output_path = BASE_DIR / "efficientnetv2_tflite_dataset_validation_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nTFLite dataset validation results saved to:")
    print(output_path)


def main():
    single_audio_comparison()
    dataset_tflite_validation()


if __name__ == "__main__":
    main()