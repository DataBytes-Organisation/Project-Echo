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

### 8. ONNX Export for Saved EfficientNetV2 Model

An ONNX export script was created:

```text
export_saved_efficientnetv2_to_onnx.py
```

This script exports the saved EfficientNetV2 model from the current training and inference workflow. Unlike the earlier reference export script, this version loads the trained checkpoint from:

```text
_trained_models/efficientnetv2_project_echo.pt
```

The script rebuilds the EfficientNetV2 architecture using the saved checkpoint information, loads the trained weights, and exports the model to ONNX using the expected input shape:

```text
[1, 1, 128, 313]
```

The ONNX model was saved as:

```text
_trained_models/efficientnetv2_project_echo.onnx
```

Output summary:

```text
Loading saved EfficientNetV2 model...
Model loaded successfully.
Model name: efficientnetv2_rw_s
Number of classes: 123
Exporting to ONNX...
ONNX model saved to:
_trained_models/efficientnetv2_project_echo.onnx
Checking ONNX model...
Opset: 13
ONNX export completed successfully.
```

### Purpose

This step prepares the trained EfficientNetV2 model for conversion into a deployment-friendly format. ONNX acts as the intermediate format between PyTorch and TensorFlow/TFLite.

---

### 9. TFLite Conversion

A conversion script was created:

```text
convert_saved_efficientnetv2_onnx_to_tflite.py
```

This script converts the exported ONNX model into a TensorFlow SavedModel and then into a TFLite model.

The conversion flow is:

```text
Saved PyTorch EfficientNetV2 model
    ↓
ONNX model
    ↓
TensorFlow SavedModel
    ↓
TFLite model
```

The following outputs were generated:

```text
_trained_models/
    efficientnetv2_project_echo.onnx
    efficientnetv2_project_echo_saved_model/
    efficientnetv2_project_echo.tflite
```

Output summary:

```text
Loading ONNX model...
Converting ONNX to TensorFlow SavedModel...
TensorFlow SavedModel saved to:
_trained_models/efficientnetv2_project_echo_saved_model
Converting TensorFlow SavedModel to TFLite...
TFLite model saved to:
_trained_models/efficientnetv2_project_echo.tflite
TFLite conversion completed successfully. File size: 85.19 MB
```

### Purpose

This step validates that the selected EfficientNetV2 model can be converted into a TFLite format. This is important for production or edge deployment, because TFLite models are generally more suitable for lightweight inference than PyTorch models.

---

### 10. TFLite Inference Validation

A TFLite validation script was created:

```text
validate_efficientnetv2_tflite_inference.py
```

This script validates the converted TFLite model using the same class mapping and preprocessing configuration used by the PyTorch predictor.

The script performs:

1. Single audio PyTorch vs TFLite comparison.
2. TFLite input and output shape inspection.
3. TFLite prediction on a real dataset audio file.
4. Dataset-based TFLite validation using 50 labelled audio files.
5. CSV and JSON output export.

TFLite input details confirmed:

```text
Input shape: [1, 1, 128, 313]
Input dtype: float32
```

TFLite output details confirmed:

```text
Output shape: [1, 123]
Output dtype: float32
```

Single audio comparison used:

```text
Acanthiza chrysorrhoa/region_3.650-4.900.mp3
```

PyTorch result:

```text
Predicted label: Acanthiza chrysorrhoa
Confidence: 0.9337119460105896
```

TFLite result:

```text
Predicted label: Acanthiza chrysorrhoa
Confidence: 0.9337119460105896
```

Comparison summary:

```text
PyTorch label: Acanthiza chrysorrhoa
TFLite label: Acanthiza chrysorrhoa
Labels match: True
PyTorch confidence: 0.9337119460105896
TFLite confidence: 0.9337119460105896
```

Dataset-based TFLite validation result:

```text
Total files tested: 50
Correct predictions: 47
TFLite validation accuracy: 0.94
```

The outputs were saved as:

```text
efficientnetv2_tflite_single_audio_comparison.json
efficientnetv2_tflite_dataset_validation_results.csv
```

### Purpose

This confirms that the converted TFLite model can run inference successfully and gives matching results with the PyTorch model for the tested sample audio. It also confirms that TFLite dataset validation matches the previous PyTorch dataset validation result of 47 correct predictions out of 50.

---

### 11. Engine-Side TFLite Message Path Validation

A safe copy of the real Engine file was created:

```text
light_echo_engine_efficientnetv2_tflite.py
```

This copied file was used so the original Engine file could remain unchanged while testing the EfficientNetV2 TFLite integration.

The copied Engine file was updated to:

1. Load the existing Engine configuration and credentials.
2. Load the EfficientNetV2 TFLite model.
3. Load the class mapping file.
4. Load the preprocessing configuration file.
5. Preprocess raw audio bytes using the EfficientNetV2 settings.
6. Run local TFLite inference inside the Engine file.
7. Add a new `EfficientNetV2_TFLite_Mode` path inside `on_message()`.

The Engine copy successfully loaded the EfficientNetV2 TFLite model.

Model loading output:

```text
EfficientNetV2 TFLite model loaded successfully.
EfficientNetV2 input shape: [  1   1 128 313]
EfficientNetV2 output shape: [  1 123]
```

The local audio-bytes inference function was tested using:

```text
Acanthiza chrysorrhoa/region_3.650-4.900.mp3
```

Local audio-bytes inference result:

```text
Predicted class: Acanthiza chrysorrhoa
Predicted probability: 93.37
Sample rate: 32000
Processed audio length: 160000
```

A fake MQTT message was then created and passed directly into `on_message()` to simulate the Engine receiving an MQTT audio message.

Fake MQTT message test result:

```text
Recieved audio message, processing via engine model...
2026-04-30T10:00:00
EfficientNetV2_TFLite_Mode
Predicted class : Acanthiza chrysorrhoa
Predicted probability : 93.37
```

Top prediction:

```text
Acanthiza chrysorrhoa
Confidence: 0.9337119460105896
```

The final API send step failed locally because `ts-api-cont` is a Docker service name and is not available when running from Jupyter.

Local API error:

```text
HTTPConnectionPool(host='ts-api-cont', port=9000): Max retries exceeded with url: /engine/event
```

This does not affect the model inference result because the prediction was already completed successfully before the API send step.

### Purpose

This validates that the copied real Engine message flow can call the EfficientNetV2 TFLite inference path from an MQTT-style payload. It confirms that raw audio from the message can be decoded, preprocessed, passed through the local TFLite model, and converted into a predicted species label with confidence.

---

## Validation Flow Completed

The completed validation flow is:

```text
Training notebook
    ↓
Saved EfficientNetV2 PyTorch model
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
    ↓
ONNX export
    ↓
TensorFlow SavedModel conversion
    ↓
TFLite conversion
    ↓
TFLite inference validation
    ↓
Engine-side EfficientNetV2 TFLite inference path
    ↓
Fake MQTT message validation through on_message()
```

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
ONNX export
TensorFlow SavedModel conversion
TFLite conversion
TFLite inference validation
Engine-side EfficientNetV2 TFLite inference path
Fake MQTT message validation through on_message()
```

Not completed yet:

```text
Run the copied Engine inside the full local Docker/system environment
Send a real MQTT audio message using EfficientNetV2_TFLite_Mode
Confirm the prediction result is sent successfully to the API/database
Decide whether to merge the EfficientNetV2 TFLite path into the original Engine file
```

## Important Notes

The current work focuses on creating a functioning Engine-compatible inference pipeline. Model accuracy can be improved later by increasing training epochs, tuning parameters, adding better validation sampling, or using more production-like audio.

The dataset validation currently tested 50 files. The first validation scan mainly covered the first dataset folders, so future validation can be improved by randomly sampling files across more species classes.

The TFLite model was successfully generated with a file size of 85.19 MB. Because model artifacts such as `.pt`, `.onnx`, `.tflite`, and TensorFlow SavedModel folders are large, the team should confirm whether these should be committed directly to GitHub or managed using Git LFS or external storage.

The current Engine-side integration was tested using a safe copied Engine file, `light_echo_engine_efficientnetv2_tflite.py`, so the original `light_echo_engine.py` remains unchanged.

The copied Engine successfully loaded the EfficientNetV2 TFLite model and processed a fake MQTT-style message through `on_message()`. The model inference completed successfully, but the final API send step failed locally because `ts-api-cont` is only available inside the Docker/system network.

The EfficientNetV2 TFLite path currently uses a new message mode called `EfficientNetV2_TFLite_Mode`. Any real MQTT message used for full-system testing must include this value in the `audioFile` field.

The next validation should be completed inside the full local system or Docker environment so the Engine can access MQTT, API, and database services properly.

---
