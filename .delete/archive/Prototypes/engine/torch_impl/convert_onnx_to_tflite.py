import onnx
import onnx2tf
import tensorflow as tf
import os

# Path to your ONNX model
ONNX_MODEL_PATH = "/Users/dinalfernando/Project-Echo/src/Prototypes/engine/torch_impl/model/cnn14/cnn14.onnx"
TFLITE_MODEL_PATH = "/Users/dinalfernando/Project-Echo/src/Prototypes/engine/torch_impl/model/cnn14/cnn14.tflite"
TF_SAVED_MODEL_PATH = "/Users/dinalfernando/Project-Echo/src/Prototypes/engine/torch_impl/model/cnn14/cnn14_savedmodel"

def convert_onnx_to_tflite():
    try:
        print("=" * 60)
        print("ONNX to TFLite Conversion")
        print("=" * 60)

        # Step 1: Validate ONNX model
        print(f"\n[1/2] Loading ONNX model from: {ONNX_MODEL_PATH}")
        onnx_model = onnx.load(ONNX_MODEL_PATH)
        onnx.checker.check_model(onnx_model)
        print("[1/2] ✓ ONNX model is valid")

        # Step 2: Convert directly to TFLite via onnx2tf
        print(f"\n[2/2] Converting ONNX → TFLite...")
        onnx2tf.convert(
            input_onnx_file_path=ONNX_MODEL_PATH,
            output_folder_path=TF_SAVED_MODEL_PATH,
            non_verbose=True,
        )
        print(f"[2/2] ✓ Done. TFLite file is inside: {TF_SAVED_MODEL_PATH}")

        # Find and report the .tflite file
        import glob
        tflite_files = glob.glob(f"{TF_SAVED_MODEL_PATH}/*.tflite")
        if tflite_files:
            print(f"\nTFLite model: {tflite_files[0]}")
            size = os.path.getsize(tflite_files[0]) / (1024 * 1024)
            print(f"Size: {size:.2f} MB")
        
    except Exception as e:
        print(f"\n✗ Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    convert_onnx_to_tflite()