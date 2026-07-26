# -*- coding: utf-8 -*-
"""Video-only inference demo for bird classifier TFLite model."""
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\jblair\.spyder-py3\birds_dataset2\bird_classifier_int8_tfcompat.tflite"
VIDEO_SOURCE_PATH = r"birds.mp4"
CLASS_NAMES = ["wedge_tailed_eagle","hardhead","cockatoo","galah","bush_stone_curlew"]
SAVE_OUTPUT = False
OUTPUT_VIDEO_PATH = r"C:\temp\classified_video.avi"
FRAME_STRIDE = 12
OVERLAY_DURATION_SEC = 3.0
CONF_THRESHOLD = 0.7
WINDOW_NAME = "Video classification"
# ---------------- LOAD TFLITE MODEL ----------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]["index"]
output_index = output_details[0]["index"]
_, height, width, _ = input_details[0]["shape"]
print("Model loaded from:", MODEL_PATH)
print("Input size expected:", (height, width))
def preprocess_rgb_array(rgb_arr):
    img = Image.fromarray(rgb_arr).resize((width, height), Image.LANCZOS)
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    input_dtype = input_details[0]["dtype"]
    if input_dtype == np.float32:
        arr = arr.astype(np.float32)
    elif input_dtype == np.uint8:
        arr = arr.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported input dtype: {input_dtype}")
    return arr
def classify_input(input_data):
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)[0]
    pred_idx = int(np.argmax(output_data))
    confidence = float(output_data[pred_idx])
    label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
    return label, confidence, pred_idx
def run_inference_on_video_file(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video file:", video_path)
        return
    out = None
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))
        print("Saving output video to:", OUTPUT_VIDEO_PATH)
    print("Starting video inference on file:", video_path)
    print("Press 'q' in the video window to quit.")
    frame_idx = 0
    current_text = None
    overlay_until = 0.0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break
        frame_idx += 1
        now = time.time()
        if FRAME_STRIDE <= 1 or frame_idx % FRAME_STRIDE == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_data = preprocess_rgb_array(frame_rgb)
            label, confidence, _ = classify_input(input_data)
            if confidence >= CONF_THRESHOLD and now >= overlay_until:
                text = f"{label} ({confidence:.2f})"
                current_text = text
                overlay_until = now + OVERLAY_DURATION_SEC
                print("Detected:", text)
        if current_text is not None and now < overlay_until:
            cv2.putText(frame_bgr, current_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            current_text = None
        cv2.imshow(WINDOW_NAME, frame_bgr)
        if SAVE_OUTPUT and out is not None:
            out.write(frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("Video inference stopped.")
if __name__ == "__main__":
    run_inference_on_video_file(VIDEO_SOURCE_PATH)
