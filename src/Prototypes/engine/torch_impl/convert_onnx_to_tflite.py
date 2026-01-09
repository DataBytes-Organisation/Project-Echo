import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os

# Path to your ONNX model
ONNX_MODEL_PATH = ".onnx"       # Your existing ONNX model
TF_SAVED_MODEL_PATH = "EFF2_QAT_Circle_clean_savedmodel"  # Temporary path for TensorFlow SavedModel
TFLITE_MODEL_PATH = "EFF2_QAT_Circle_clean.tflite" # Output TFLite model

def convert_onnx_to_tflite():
    """
    Convert ONNX model to TFLite format
    """
    try:
        print("=" * 60)
        print("ONNX to TFLite Conversion")
        print("=" * 60)
        
        # Step 1: Load ONNX model
        print(f"\n[1/4] Loading ONNX model from: {ONNX_MODEL_PATH}")
        onnx_model = onnx.load(ONNX_MODEL_PATH)
        
        # Check and simplify the model
        print("[1/4] Checking ONNX model...")
        onnx.checker.check_model(onnx_model)
        print("[1/4]  ONNX model is valid")
        
        # Step 2: Convert ONNX to TensorFlow
        print(f"\n[2/4] Converting ONNX to TensorFlow SavedModel...")
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(TF_SAVED_MODEL_PATH)
        print(f"[2/4]  TensorFlow SavedModel saved to: {TF_SAVED_MODEL_PATH}")
        
        # Step 3: Load TensorFlow SavedModel
        print(f"\n[3/4] Loading TensorFlow SavedModel...")
        converter = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL_PATH)
        
        # Use minimal conversion settings to preserve model accuracy
        # Don't apply additional optimizations since the model is already trained
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        print("[3/4]  Converter configured")
        
        # Step 4: Convert to TFLite
        print(f"\n[4/4] Converting to TFLite format...")
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        
        print(f"[4/4]  TFLite model saved to: {TFLITE_MODEL_PATH}")
        
        # Display file sizes
        print("\n" + "=" * 60)
        print("Conversion Summary")
        print("=" * 60)
        onnx_size = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
        tflite_size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
        print(f"ONNX model size:   {onnx_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
        print(f"Size reduction:    {((onnx_size - tflite_size) / onnx_size * 100):.1f}%")
        print("=" * 60)
        print(" Conversion completed successfully!")
        
    except FileNotFoundError:
        print(f"\n Error: Could not find ONNX model at: {ONNX_MODEL_PATH}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"\n Error during conversion: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure your ONNX model is valid")
        print("2. Check that all dependencies are installed correctly")
        raise

if __name__ == "__main__":
    convert_onnx_to_tflite()