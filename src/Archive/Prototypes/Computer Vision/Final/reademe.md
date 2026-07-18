# Bird / Creature Image Classification Pipeline

This repository contains a full end-to-end workflow for building an image-classification model for creatures (e.g., birds). It is organised into **three folders**, each representing a major stage in the pipeline:

1. **Create Dataset** – Automatically collect and augment creature images  
2. **Train Model** – Train a TensorFlow classifier and export a quantized TFLite model  
3. **Inference** – Run classification on images or video using the exported model  

---

# 1. Folder Structure

project/
¦
+-- Create Dataset/
¦   +-- get_creatures_v2.py
¦   +-- augment_creatures_v2.py
¦   +-- creatures.json
¦   +-- How to Create A Creatures Dataset_v2.docx
¦
+-- Train Model/
¦ 	+-- train_model_v1.py
¦   +-- requirements.txt
¦   +-- class_names.csv
¦
¦
+-- Inference/
    +-- classify_video_v1.py
    +-- classify_stills_v1.py
	+-- requirements.txt
    +-- class_names.csv

---

# 2. CREATE DATASET

## 2.1 get_creatures_v2.py – Download & standardise images

Run locally - Reads creatures.json and for each class, downloads images from Wikimedia Commons, converts them to 300×300 RGB, and outputs folder with class name.

Run:
```
python get_creatures_v2.py
```

Copy folders to Google Drive.
---

## 2.2 augment_creatures_v1.py – Dataset expansion

Creates 500 augmented images for each class using flips, rotation, blur, noise and colour shifts.

Run script in Google Colab,
---

## 2.3 Documentation

“How to Create A Creatures Dataset_v1.docx”  
Explains dataset creation in non-technical language.

---

# 3. TRAIN MODEL

## train_model_v1.ipynb – Train MobileNetV2 + export TFLite

Trains a deep-learning classifier in Google Colab and exports:

- TensorFlow SavedModel  
- Quantized TFLite model  
- Accuracy/loss plots
- saves class names in training order to class_names.csv - this is essential for inference.

---

# 4. INFERENCE

## classify_stills_v2.py – Video classification

Loads the TFLite model and runs classification on still images. Class_names.csv is required for correct output of class names.

Run:
```
python classify_video_v2.py
```

Press **Q** to quit window.

---

## classify_video_v2.py – Video classification

Loads the TFLite model and runs classification on frames from a video file or camera feed.  Class_names.csv is required for correct output of class names.

Run:
```
python classify_video_v2.py
```

Press **Q** to quit window.
# 5. Pipeline Summary

STEP 1 — Dataset  
- Run get_creatures  
- Run augment_creatures  

STEP 2 — Training  
- Run train_model  

STEP 3 — Inference  
- Update MODEL_PATH  
- Run classify_video or Classify_stills 

---

# 6. Notes

- Class_names.csv (Created from training script) is required for inference. 
- training 50 classes on Google Colab with A100 GPUs takes < 10 mins
- TFLite is for inference only  

---

# 7. Troubleshooting

Wrong predictions ? class order mismatch  
Video slow ? increase FRAME_STRIDE  
“No images found” ? check folder paths

---

# 8. Credits

Scripts by James Blair (2026).  
Images from Wikimedia Commons (licensing varies).
