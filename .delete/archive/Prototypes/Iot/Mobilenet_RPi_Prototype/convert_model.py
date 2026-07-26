#!/usr/bin/env python3
"""
Model conversion utility.

This module converts a Keras H5 model file to TensorFlow Lite (.tflite) format.
It loads the H5 model, converts it using the TensorFlow converter, writes the
TFLite file to disk and performs a small inference test using dummy data.

"""

import os
import sys
import argparse
import numpy as np


def convert_h5_to_tflite(h5_path, tflite_path=None):
    """Convert a Keras H5 model to TensorFlow Lite format.

    Args:
        h5_path (str): Path to the input Keras H5 model file.
        tflite_path (str|None): Optional path for the output TFLite file. If
            omitted the function will replace the H5 extension with .tflite.

    Returns:
        bool: True if conversion and test succeeded, False otherwise.

    The function will attempt to import TensorFlow and, if available,
    will also import tensorflow_hub to support models that use KerasLayer.
    If tensorflow_hub is not present the loader will attempt to load the
    model without custom objects. Any errors are printed and the function
    returns False.
    """

    # Import TensorFlow lazily so the module can be used without TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("TensorFlow is required for model conversion")
        return False

    # Determine the default output path if not supplied.
    if tflite_path is None:
        tflite_path = h5_path.replace('.hdf5', '.tflite').replace('.h5', '.tflite')

    print("Converting model")
    print(f"  Input:  {h5_path}")
    print(f"  Output: {tflite_path}")

    try:
        # Load the Keras H5 model. Require tensorflow_hub so models that
        # embed hub layers are loaded with the correct custom object.
        print("Loading H5 model")
        import tensorflow_hub as hub
        with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}): 
            model = tf.keras.models.load_model(h5_path) 

        # For debugging purposes, report basic model information.
        print("Model loaded successfully")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Parameters: {model.count_params():,}")

        # Create the TFLite converter from the Keras model object.
        print("Creating TFLite converter")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Perform the conversion to TFLite bytes.
        print("Converting to TFLite format")
        tflite_model = converter.convert()

        # Write the converted model to disk.
        print("Saving TFLite model to disk")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        # Compute and display file size information to help with verification.
        h5_size = os.path.getsize(h5_path) / (1024 * 1024)
        tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
        compression_ratio = (h5_size - tflite_size) / h5_size * 100

        print("Conversion completed successfully")
        print("Model size comparison:")
        print(f"  Original H5: {h5_size:.1f} MB")
        print(f"  TFLite:      {tflite_size:.1f} MB")
        print(f"  Compression: {compression_ratio:.1f}% smaller")

        # Debug step
        # Run a small test on the converted model using dummy input to ensure
        # that the interpreter loads and produces outputs in the expected
        # shape and range.
        print("Testing converted model")
        test_tflite_model(tflite_path, model.input_shape[1:])

        return True

    except Exception as e:
        # Any exception during conversion or saving will be printed and the
        # function returns False to indicate failure.
        print(f"Conversion failed: {e}")
        return False


def test_tflite_model(tflite_path, input_shape):
    """Test a TFLite model by running a single inference with dummy data.

    Args:
        tflite_path (str): Path to the TFLite model file to test.
        input_shape (tuple): Shape of the model input tensor (excluding batch
            dimension), for example (224, 224, 3).

    Returns:
        bool: True if the test inference succeeds, False otherwise.

    The function will attempt to import the lightweight tflite_runtime
    interpreter first. If that is not installed it will fall back to the
    TensorFlow bundled interpreter. A random dummy input is created to match
    the interpreter's expected input shape and a single invoke is performed.
    """

    try:
        # Prefer the smaller tflite_runtime package when available.
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite

        # Load the TFLite model into an interpreter and allocate tensors.
        interpreter = tflite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Query input and output tensor details for verification.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"  Input details: {input_details[0]['shape']}")
        print(f"  Output details: {output_details[0]['shape']}")

        # Create a dummy input array matching the interpreter's expected shape
        # and data type. Use a random float array within [0, 1).
        dummy_input = np.random.random(input_details[0]['shape']).astype(np.float32)

        # Perform inference.
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()

        # Retrieve the output tensor and report basic statistics.
        output_data = interpreter.get_tensor(output_details[0]['index'])

        print("TFLite model test successful")
        print(f"  Output shape: {output_data.shape}")
        print(f"  Output range: [{output_data.min():.3f}, {output_data.max():.3f}]")

        return True

    except Exception as e:
        print(f"TFLite model test failed: {e}")
        return False


def main():
    """Command line entry point for the conversion utility.

    This function parses command line arguments and invokes the conversion
    routine. If no arguments are provided and a default model is present at
    Model/Model.h5 the function will attempt to convert that file.
    """

    parser = argparse.ArgumentParser(description='Convert Keras H5 model to TensorFlow Lite')
    parser.add_argument('input', nargs='?', help='Input H5 model file path')
    parser.add_argument('-o', '--output', help='Output TFLite model file path')

    args = parser.parse_args()

    # If the user did not provide an input path, try to use the default model
    # path used in this project.
    if not args.input:
        default_h5 = "Model/Model.h5"
        if os.path.exists(default_h5):
            print("No arguments provided; converting default model")
            success = convert_h5_to_tflite(default_h5, args.output)
            sys.exit(0 if success else 1)
        else:
            print(f"Input file not provided and default model not found: {default_h5}")
            print("Usage: python convert_model.py <input_h5_file> [-o output_tflite_file]")
            sys.exit(1)

    # Validate the provided input file path exists and appears to be H5 format.
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    if not (args.input.endswith('.h5') or args.input.endswith('.hdf5')):
        print("Input file must be in H5 format (.h5 or .hdf5)")
        sys.exit(1)

    # Convert the model and report the overall result.
    print("TensorFlow Lite model converter")
    print("=" * 40)

    success = convert_h5_to_tflite(args.input, args.output)

    if success:
        print("Conversion completed successfully")
        print("You can now use the TFLite model for faster inference.")
    else:
        print("Conversion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
