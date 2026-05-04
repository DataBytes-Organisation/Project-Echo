"""
Convert the saved Project Echo EfficientNetV2 ONNX model to TFLite.

This script converts:
PyTorch saved model -> ONNX -> TensorFlow SavedModel -> TFLite

The ONNX model should already be created by:
export_saved_efficientnetv2_to_onnx.py
"""

import pathlib
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare


BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "_trained_models"

ONNX_MODEL_PATH = MODEL_DIR / "efficientnetv2_project_echo.onnx"
SAVED_MODEL_DIR = MODEL_DIR / "efficientnetv2_project_echo_saved_model"
TFLITE_MODEL_PATH = MODEL_DIR / "efficientnetv2_project_echo.tflite"


def convert_onnx_to_saved_model():
    print("Loading ONNX model...")
    onnx_model = onnx.load(ONNX_MODEL_PATH)

    print("Converting ONNX to TensorFlow SavedModel...")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(SAVED_MODEL_DIR))

    print("TensorFlow SavedModel saved to:")
    print(SAVED_MODEL_DIR)


def convert_saved_model_to_tflite():
    print("Converting TensorFlow SavedModel to TFLite...")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))

    # Basic float32 TFLite conversion first.
    # Quantisation can be added later if needed.
    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print("TFLite model saved to:")
    print(TFLITE_MODEL_PATH)


def main():
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL_PATH}")

    convert_onnx_to_saved_model()
    convert_saved_model_to_tflite()

    if TFLITE_MODEL_PATH.exists():
        size_mb = TFLITE_MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"TFLite conversion completed successfully. File size: {size_mb:.2f} MB")
    else:
        raise RuntimeError("TFLite file was not created.")


if __name__ == "__main__":
    main()