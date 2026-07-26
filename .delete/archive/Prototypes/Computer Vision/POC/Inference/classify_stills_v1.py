# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 21:25:10 2025

@author: jblair

This script classifies still images for the below classes of bird.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- CONFIG ----------------

# MODEL_PATH = "bird_classifier_mobilenetv2_300_quant2.tflite"
MODEL_PATH = r"C:\Users\jblair.spyder-py3\birds_dataset2\bird_classifier_int8_tfcompat.tflite"

# Folder with images to classify (change this to your folder)
IMAGE_DIR = r"C:\Users\jblair.spyder-py3\birds_dataset2\testing"

# IMPORTANT: class order must match training order.
# Get this from the Classes print out when training is complete.
CLASS_NAMES = [
    "wedge_tailed_eagle",   # class 0 aquila audax
    # "dusky_woodswallow",  # class 1 artamus cyanopterus - removed
    "hardhead",             # class 1 aythya australi
    "cockatoo",             # class 2 cacatua galerita
    "galah",                # class 3 eolophus roseicapilla
    "bush_stone_curlew",    # class 4 burhinus grallarius
]

# ---------------- LOAD TFLITE MODEL ----------------

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

# Get expected input size from model
_, height, width, _ = input_details[0]["shape"]

print("Model loaded from:", MODEL_PATH)
print("Input size expected:", (height, width))


# ---------------- IMAGE PREP FUNCTION ----------------

def load_and_preprocess_image(path):
    """Load image from disk, resize, and format for TFLite model."""
    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    arr = np.array(img)

    # Add batch dimension: [1, h, w, 3]
    arr = np.expand_dims(arr, axis=0)

    # Cast to the dtype the model expects
    input_dtype = input_details[0]["dtype"]
    if input_dtype == np.float32:
        arr = arr.astype(np.float32)
    elif input_dtype == np.uint8:
        arr = arr.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported input dtype: {input_dtype}")

    return arr


# ---------------- INFERENCE LOOP ----------------

def run_inference_on_folder(folder):
    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print("No images found in", folder)
        return

    print(f"Found {len(image_files)} images in {folder}")

    for fname in sorted(image_files):
        fpath = os.path.join(folder, fname)

        # Prepare input
        input_data = load_and_preprocess_image(fpath)

        # Set tensor and invoke
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        # Get prediction
        output_data = interpreter.get_tensor(output_index)[0]  # shape [num_classes]
        pred_idx = int(np.argmax(output_data))
        confidence = float(output_data[pred_idx])

        label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
        print(f"{fname:40s} -> {label:20s} (conf: {confidence:.3f})")


# ---------------- RUN ----------------

if __name__ == "__main__":
    run_inference_on_folder(IMAGE_DIR)
