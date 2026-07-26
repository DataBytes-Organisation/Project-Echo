# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 11:46:38 2025

@author: james
"""

import os
import csv
import re
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\james\.spyder-py3\bird_classifier_float.tflite"
CSV_PATH   = r"C:\Users\james\.spyder-py3\class_names.csv"
VIDEO_PATH = r"C:\Users\james\.spyder-py3\birds.mp4"

IMG_SIZE = 300

FRAME_STRIDE = 36
DISPLAY_SECONDS = 2.0
WINDOW_NAME = "Bird Classifier (TFLite)"

ACCURACY_THRESHOLD = 60.0   # percent (e.g. 70.0 = only show/save ≥70%)

OUTPUT_DIR = r"C:\Users\james\.spyder-py3"
VIDEO_SAVE_DIR = r"C:\temp"  # <-- save annotated video copy here

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PRED_CSV_PATH = os.path.join(OUTPUT_DIR, f"video_predictions_{TIMESTAMP}.csv")
OUT_VIDEO_PATH = os.path.join(VIDEO_SAVE_DIR, f"annotated_video_{TIMESTAMP}.mp4")

# ---------------- LOAD MODEL ----------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_index  = input_details[0]["index"]
out_index = output_details[0]["index"]

# ---------------- LOAD CLASSES ----------------
class_names = []
with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
    for row in csv.reader(f):
        if row:
            class_names.append(row[0])

# ---------------- HELPERS ----------------
def preprocess_bgr_frame(frame_bgr: np.ndarray) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb).resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img, dtype=np.float32)
    return np.expand_dims(x, axis=0)

def softmax_if_needed(raw: np.ndarray) -> np.ndarray:
    raw = raw.astype(np.float32)
    s = float(raw.sum())
    if raw.min() >= 0.0 and raw.max() <= 1.0 and 0.95 <= s <= 1.05:
        return raw
    e = np.exp(raw - np.max(raw))
    return e / np.sum(e)

def clean_label(label: str) -> str:
    # Remove trailing image-size suffixes like _300x300, _300x300a, etc.
    return re.sub(r"_\d+x\d+[a-zA-Z]*$", "", label)


def draw_highlighted_text(
    img: np.ndarray,
    text: str,
    org: tuple,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    thickness: int = 2,
    pad: int = 8,
):
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    x0 = x - pad
    y0 = y - th - pad
    x1 = x + tw + pad
    y1 = y + baseline + pad

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img.shape[1] - 1, x1)
    y1 = min(img.shape[0] - 1, y1)

    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), -1)  # green highlight
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)  # black text

# ---------------- VIDEO ----------------
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

writer_video = None

frame_idx = -1
last_text = ""
last_text_until_sec = -1.0

with open(PRED_CSV_PATH, "w", newline="", encoding="utf-8") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow([
        "run_timestamp",
        "video_path",
        "frame_index",
        "video_time_sec",
        "label",
        "prob_percent"
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if writer_video is None:
            if w <= 0 or h <= 0:
                h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_video = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, float(fps), (w, h))

        frame_idx += 1
        video_time_sec = frame_idx / fps

        if FRAME_STRIDE > 0 and frame_idx % FRAME_STRIDE == 0:
            input_data = preprocess_bgr_frame(frame)

            interpreter.set_tensor(in_index, input_data)
            interpreter.invoke()

            output = interpreter.get_tensor(out_index)[0]
            probs = softmax_if_needed(output)

            idx = int(np.argmax(probs))
            prob = float(probs[idx]) * 100.0

            if prob >= ACCURACY_THRESHOLD:
                raw_label = class_names[idx] if idx < len(class_names) else str(idx)
                label = clean_label(raw_label)

                last_text = f"{label}  {prob:.1f}%"
                last_text_until_sec = video_time_sec + DISPLAY_SECONDS

                writer.writerow([
                    TIMESTAMP,
                    VIDEO_PATH,
                    frame_idx,
                    f"{video_time_sec:.3f}",
                    label,
                    f"{prob:.2f}"
                ])

        if video_time_sec <= last_text_until_sec and last_text:
            draw_highlighted_text(frame, last_text, (20, 50))

        cv2.imshow(WINDOW_NAME, frame)
        if writer_video is not None:
            writer_video.write(frame)

        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break

cap.release()
if writer_video is not None:
    writer_video.release()
cv2.destroyAllWindows()

print("Predictions saved to:", PRED_CSV_PATH)
print("Annotated video saved to:", OUT_VIDEO_PATH)
