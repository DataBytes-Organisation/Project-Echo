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
¦   +-- get_creatures_v1.py
¦   +-- augment_creatures_v1.py
¦   +-- How to Create A Creatures Dataset_v1.docx
¦
+-- Train Model/
¦   +-- Computer Vision on Edge Devices_v1.docx
¦ 	+-- train_model_v1.py
¦   +-- requirements.txt
¦
+-- Inference/
    +-- classify_video_v1.py
    +-- classify_stills_v1.py
	+-- requirements.txt

---

# 2. CREATE DATASET

## 2.1 get_creatures_v1.py – Download & standardise images

Downloads images from Wikimedia Commons, converts them to 300×300 RGB, and outputs a ZIP + folder.

Run:
```
python get_creatures_v1.py
```

---

## 2.2 augment_creatures_v1.py – Dataset expansion

Creates thousands of augmented images using flips, rotation, blur, noise and colour shifts.

Run:
```
python augment_creatures_v1.py
```

---

## 2.3 Documentation

“How to Create A Creatures Dataset_v1.docx”  
Explains dataset creation in non-technical language.

---

# 3. TRAIN MODEL

## train_model_v1.py – Train MobileNetV2 + export TFLite

Trains a deep-learning classifier and exports:

- TensorFlow SavedModel  
- Quantized TFLite model  
- Accuracy/loss plots

Run:
```
python train_model_v1.py
```

---

## 3.1 Documentation

“Computer Vision on Edge Devices_v1.docx”  
Explains why the current pipeline has been selected.

# 4. INFERENCE

## classify_stills_v1.py – Video classification

Loads the TFLite model and runs classification on still images.

Run:
```
python classify_video_v1.py
```

Press **Q** to quit window.

---

## classify_video_v1.py – Video classification

Loads the TFLite model and runs classification on frames from a video file or camera.

Run:
```
python classify_video_v1.py
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

- Class order must match training folder order  
- a 5 classes model with 1,000 images a class takes roughly 50 mins to train on CPU (i7 with 32 GB RAM) 
- TFLite is for inference only  

---

# 7. Troubleshooting

Wrong predictions ? class order mismatch  
Video slow ? increase FRAME_STRIDE  
“No images found” ? check folder paths

---

# 8. Credits

Scripts by James Blair (2025).  
Images from Wikimedia Commons (licensing varies).
