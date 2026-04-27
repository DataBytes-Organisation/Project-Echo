# EfficientNetV2 Engine Integration Progress

## Overview

This work integrates the EfficientNetV2 model selected from the Sprint 1 benchmarking work into a reusable inference workflow for Project Echo. The main goal of this part is to create a functioning model inference pipeline that can load a saved model, preprocess audio, return species predictions, and support future Engine and MQTT integration.

At this stage, the focus is not on maximising model accuracy. The priority is to prove that the selected model can be trained, saved, reloaded, and used in a structured Engine-style inference flow.

---

## Branch

Work was completed on the branch:

```text
EE/WKF/efficientnetv2_integration
```

---

## Work Completed So Far

### 1. EfficientNetV2 Training and Model Saving

A training notebook was created to retrain the selected EfficientNetV2 model using the available Project Echo dataset.

The notebook performs the following steps:

1. Loads the audio dataset from the local data folder.
2. Extracts Mel spectrogram features from audio files.
3. Trains the EfficientNetV2 model using the extracted features.
4. Evaluates the trained model quickly on test data.
5. Saves the trained model and supporting configuration files.

The following output files are generated:

```text
_trained_models/
    efficientnetv2_project_echo.pt
    class_mapping.json
    preprocess_config.json
    training_metrics.json
```

### Purpose

This step creates the saved model file required for Engine-side inference.

---

### 2. Saved Model Loading Test

After training, the saved model was loaded again to confirm that it can be reused outside the training process.

This test confirms:

1. The `.pt` model file loads successfully.
2. The model architecture can be rebuilt using the saved checkpoint information.
3. The saved class mapping can be used to convert predicted class indexes into species labels.
4. A sample audio file can be passed through the loaded model.
5. The model returns a predicted label and confidence value.

### Purpose

This proves that the saved model is usable for inference and is not only valid inside the training notebook.

---

### 3. Reusable EfficientNetV2 Predictor

A reusable Python predictor file was created:

```text
efficientnetv2_predictor.py
```

This file contains reusable functions for model inference:

```text
load_model()
load_class_mapping()
load_preprocess_config()
preprocess_audio()
predict_audio()
```

The predictor handles:

1. Loading the saved EfficientNetV2 model.
2. Loading the class mapping file.
3. Loading the preprocessing configuration file.
4. Loading and padding/trimming audio.
5. Extracting Mel spectrogram features.
6. Running model inference.
7. Returning the predicted label, class index, confidence value, and top predictions.

### Purpose

This separates inference logic from the notebook, making the code reusable by the real Engine pipeline later.

---

### 4. Single Audio Inference Test

The reusable predictor was tested using a real dataset audio file:

```text
Acanthiza chrysorrhoa/region_3.650-4.900.mp3
```

The model predicted:

```text
Predicted label: Acanthiza chrysorrhoa
Confidence: 0.9337
```

The top prediction matched the actual folder label.

### Purpose

This confirms that the reusable predictor can correctly process a real audio file and return a valid prediction.

---

### 5. Dataset-Based Validation

A dataset validation script was created:

```text
validate_efficientnetv2_dataset_inference.py
```

This script validates the saved EfficientNetV2 model using labelled dataset audio files.

It checks:

1. The saved model loads correctly.
2. The class mapping loads correctly.
3. The preprocessing config loads correctly.
4. Dataset audio files can be preprocessed.
5. The model returns predicted labels and confidence values.
6. Predictions can be compared with actual dataset folder labels.

The validation tested 50 audio files.

Result:

```text
Total files tested: 50
Correct predictions: 47
Validation accuracy: 0.94
```

The validation output was saved as:

```text
efficientnetv2_dataset_validation_results.csv
```

### Purpose

This answers the dataset validation part of the task. It proves that the model can run inference on labelled dataset audio and return labels and confidence values.

---

### 6. JSON-Style Inference Validation

A JSON-style validation script was created:

```text
validate_efficientnetv2_json_inference.py
```

This script simulates the type of structured input that the Engine may receive before or through MQTT.

Example input:

```json
{
    "device_id": "sensor_001",
    "audio_path": "path/to/audio.mp3",
    "timestamp": "2026-04-27T15:30:00",
    "source": "dataset_validation"
}
```

The script:

1. Reads the JSON-style message.
2. Extracts the audio path.
3. Validates that the audio file exists.
4. Runs preprocessing and inference.
5. Returns a structured JSON-style prediction response.

Example output includes:

```json
{
    "device_id": "sensor_001",
    "model": "EfficientNetV2",
    "prediction": {
        "label": "Acanthiza chrysorrhoa",
        "class_index": 0,
        "confidence": 0.9337119460105896
    },
    "status": "success"
}
```

The result was saved as:

```text
efficientnetv2_json_inference_result.json
```

### Purpose

This proves that the model can work with a structured JSON-style Engine input, which is the step before real MQTT integration.

---

### 7. MQTT Message Handler Validation

A simulated MQTT message handler validation script was created:

```text
validate_efficientnetv2_mqtt_message_handler.py
```

This script does not require a live MQTT broker. Instead, it simulates how an MQTT payload would arrive as bytes.

The simulated MQTT payload is:

```text
MQTT payload bytes → decode JSON → extract audio path → run predictor → return prediction response
```

The script validates:

1. MQTT-style payload bytes can be decoded.
2. JSON fields are read correctly.
3. The audio path is extracted correctly.
4. The EfficientNetV2 model loads successfully.
5. Audio preprocessing runs before inference.
6. Label, confidence, class index, and top predictions are returned.
7. The response is saved as JSON.

The result was saved as:

```text
efficientnetv2_mqtt_message_handler_result.json
```

### Purpose

This prepares the model inference flow for real MQTT integration later.

---

## Validation Flow Completed

The completed validation flow is:

```text
Training notebook
    ↓
Saved EfficientNetV2 model
    ↓
Reusable predictor file
    ↓
Single audio inference test
    ↓
Dataset validation
    ↓
JSON-style validation
    ↓
MQTT message handler validation
```

---

## Current Status

Completed:

```text
EfficientNetV2 training and saving
Saved model loading test
Reusable predictor implementation
Single audio inference validation
Dataset validation
JSON-style inference validation
MQTT message handler validation
```

Not completed yet:

```text
Direct connection to the real Engine pipeline
Live MQTT broker integration
```

---

## Important Notes

The current work focuses on creating a functioning Engine-compatible inference pipeline. Model accuracy can be improved later by increasing training epochs, tuning parameters, adding better validation sampling, or using more production-like audio.

The dataset validation currently tested 50 files. The first validation scan mainly covered the first dataset folders, so future validation can be improved by randomly sampling files across more species classes.

The `.pt` model file may be large, so it should not be committed to GitHub unless the team confirms that model files should be stored in the repository. If needed, model files can be shared through external storage or Git LFS.

---
