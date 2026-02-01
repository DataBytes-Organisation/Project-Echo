# Project Echo Engine - Complete Documentation

## Table of Contents

1. Project Overview
2. Quick Start (For Juniors)
3. Technical Deep Dive (For Seniors)
4. Architecture & Components
5. Configuration & Credentials
6. Deployment (Docker)
7. Edge Cases & Warnings

---

## Project Overview

Project Echo is a bioacoustics classification system designed to detect and classify noise-producing animal species in rainforest environments. Its primary goal is to provide conservationists with non-invasive tools to monitor biodiversity and animal density.

### What is the Engine?

The Engine (`src/Components/Engine`) is the production-ready core processing unit responsible for:

1. **Real-time Audio Processing**: Subscribes to MQTT topics and receives audio clips from IoT sensors
2. **Feature Extraction**: Converts raw audio into Mel spectrograms optimized for CNN input
3. **Deep Learning Inference**: Classifies species using TensorFlow models served via TensorFlow Serving
4. **Event Detection**: Supports multiple detection modes including sound event detection (YAMNet-based) and weather classification
5. **Database Integration**: Stores predictions with metadata (location, confidence, timestamps) in MongoDB

### System Architecture

```
IoT Sensors → MQTT Broker → Echo Engine → TensorFlow Serving → MongoDB
                                ↓
                          Audio Pipeline
                        (Mel Spectrograms)
```

### Codebase Structure

> **Important:** The codebase contains two distinct implementations:
> - **Stable Engine** (`src/Components/Engine`): Production-ready implementation covered in this document
> - **Prototype Engine** (`src/Prototypes/engine/torch_impl`): Experimental PyTorch-based implementation for future IoT/TFLite deployment

This documentation focuses on the **Stable Engine**.

---

## Quick Start (For Juniors)

### Prerequisites

Before running the engine, ensure your environment is set up:

| Requirement | Details |
|-------------|---------|
| OS | Linux (Ubuntu recommended), macOS, or Windows with WSL2 |
| Python | Version 3.9+ |
| System Tools | ffmpeg (crucial for audio processing) |
| Hardware | CPU sufficient for inference; NVIDIA GPU recommended for training (CUDA 11.x+) |
| Docker | Recommended for deployment |

### Installing FFmpeg

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

### Installation Steps

**1. Clone the Repository**

```bash
git clone https://github.com/DataBytes-Organisation/Project-Echo.git
cd Project-Echo
```

**2. Navigate to Engine Directory**

```bash
cd src/Components/Engine
```

**3. Install Python Dependencies**

```bash
pip install -r requirements.txt
```



**4. Authenticate with Google Cloud (for species list loading)**

```bash
gcloud auth application-default login
```

### Running the Engine Locally

The engine operates as an MQTT subscriber waiting for audio events:

```bash
python echo_engine.py
```

**Expected output:**

```
Python Version           :  3.9.x
TensorFlow Version       :  2.15.0
Librosa Version          :  0.10.x
Echo Engine configuration successfully loaded
Echo Engine credentials successfully loaded
Found echo store database names: ['EchoNet']
Engine started.
Subscribing to MQTT: ts-mqtt-server-cont projectecho/engine/2
Retrieving species names from GCP
Engine waiting for audio to arrive...
```

### Onboarding Task

For hands-on learning, complete the onboarding notebook:

```bash
cd "Tutorials/Onboarding Task"
jupyter notebook onboarding_updated.ipynb
```

This notebook will guide you through:
- Loading sample audio files
- Visualizing Mel spectrograms
- Running test inference with pre-trained models

---

## Technical Deep Dive (For Seniors)

### Core Processing Pipeline

The engine implements an ETL (Extract, Transform, Load) pattern optimized for audio tensor processing.

#### 1. Audio Ingestion & Normalization

```python
# From combined_pipeline() method
audio_clip, sample_rate = librosa.load(file, sr=self.config['AUDIO_SAMPLE_RATE'])

# Keep right channel only (if stereo)
if audio_clip.ndim == 2 and audio_clip.shape[0] == 2:
    audio_clip = audio_clip[1, :]

# Cast to float32
audio_clip = audio_clip.astype(np.float32)
```

**Key Points:**
- Default sample rate: 48kHz (configurable via `AUDIO_SAMPLE_RATE`)
- Mono conversion reduces tensor dimensionality
- All audio standardized to float32 for consistent processing

#### 2. Windowing Strategy

The engine uses random subsection sampling to analyze fixed-duration clips:

```python
def load_random_subsection(self, tmp_audio_t, duration_secs):
    audio_duration_secs = tf.shape(tmp_audio_t)[0] / self.config['AUDIO_SAMPLE_RATE']
    
    if audio_duration_secs > duration_secs:
        # Random 5-second window
        max_start = tf.cast(audio_duration_secs - duration_secs, tf.float32)
        start_time_secs = tf.random.uniform((), 0.0, max_start, dtype=tf.float32)
        start_index = tf.cast(start_time_secs * self.config['AUDIO_SAMPLE_RATE'], dtype=tf.int32)
        end_index = start_index + duration_secs * self.config['AUDIO_SAMPLE_RATE']
        subsection = tmp_audio_t[start_index:end_index]
    else:
        # Zero-pad if shorter than window
        padding_length = duration_secs * self.config['AUDIO_SAMPLE_RATE'] - tf.shape(tmp_audio_t)[0]
        padding = tf.zeros([padding_length], dtype=tmp_audio_t.dtype)
        subsection = tf.concat([tmp_audio_t, padding], axis=0)
    
    return subsection
```

**Default Window:** 5 seconds (`AUDIO_CLIP_DURATION`)

#### 3. Mel Spectrogram Generation

```python
# Compute mel-spectrogram
image = librosa.feature.melspectrogram(
    y=audio_clip, 
    sr=self.config['AUDIO_SAMPLE_RATE'],      # 48000 Hz
    n_fft=self.config['AUDIO_NFFT'],          # 2048
    hop_length=self.config['AUDIO_STRIDE'],   # 200
    n_mels=self.config['AUDIO_MELS'],         # 260
    fmin=self.config['AUDIO_FMIN'],           # 20 Hz
    fmax=self.config['AUDIO_FMAX'],           # 13000 Hz
    win_length=self.config['AUDIO_WINDOW']    # 500
)

# Convert to log scale (dB)
image = librosa.power_to_db(
    image, 
    top_db=self.config['AUDIO_TOP_DB'],       # 80 dB
    ref=1.0
)
```

**Mel Filterbank Rationale:** Maps FFT bins to the Mel scale to mimic animal hearing perception, improving classification accuracy for bioacoustic signals.

#### 4. Image Preprocessing for CNN

```python
# Reshape: (time, frequency) with 3 color channels
image = np.moveaxis(image, 1, 0)
image = tf.expand_dims(image, -1)
image = tf.repeat(image, self.config['MODEL_INPUT_IMAGE_CHANNELS'], axis=2)

# Resize to model input size (260x260)
image = tf.image.resize(
    image, 
    (self.config['MODEL_INPUT_IMAGE_WIDTH'], self.config['MODEL_INPUT_IMAGE_HEIGHT']), 
    method=tf.image.ResizeMethod.LANCZOS5  # High-quality interpolation
)

# Normalize to [0, 1]
image = image - tf.reduce_min(image)
image = image / (tf.reduce_max(image) + 0.0000001)
```

**Output Shape:** `(260, 260, 3)` - matches pre-trained CNN input expectations

#### 5. Model Inference

The engine uses TensorFlow Serving for scalable model deployment:

```python
# Prepare request
image = tf.expand_dims(image, 0)  # Add batch dimension
image_list = image.numpy().tolist()
data = json.dumps({"signature_name": "serving_default", "inputs": image_list})

# POST to TensorFlow Serving endpoint
url = self.config['MODEL_SERVER']
headers = {"content-type": "application/json"}
json_response = requests.post(url, data=data, headers=headers)
model_result = json.loads(json_response.text)
predictions = model_result['outputs'][0]

# Extract prediction
predicted_class, predicted_probability = self.predict_class(predictions)
```

**Model Server Endpoint:** `http://ts-echo-model-cont:8501/v1/models/echo_model/versions/1:predict`

### Detection Modes

The engine supports three operational modes:

#### Mode 1: Recording Mode (Classic)

- **Use Case:** Direct species classification from audio clips
- **Pipeline:** `combined_pipeline()` → TensorFlow Serving → Single prediction
- **Trigger:** `audio_event['audioFile'] == "Recording_Mode"`

#### Mode 2: Sound Event Detection (YAMNet-based)

- **Use Case:** Long audio files with multiple animal vocalizations
- **Pipeline:**
  1. YAMNet detects animal-related sounds in 1-second chunks
  2. Segments buffered when animal activity detected
  3. EchoNet model classifies each segment
  4. Returns timestamped detections with confidence scores
- **Trigger:** `audio_event['audioFile'] == "Recording_Mode_V2"`
- **Output:** DataFrame with columns: `start_time`, `end_time`, `echonet_label_1`, `echonet_confidence_1`

```python
def sound_event_detection(self, filepath, sample_rate):
    # Split into 1-second chunks
    frame_len = int(sr * 1)
    chunks = [data[i*frame_len:(i+1)*frame_len] for i in range(num_chunks)]
    
    # Detect animal sounds using YAMNet
    for cnt, frame_data in enumerate(chunks):
        outputs = yamnet(frame_data)
        yamnet_prediction = np.mean(outputs[0], axis=0)
        
        # Check for animal-related classes
        if any(yamnet_prediction[...] >= threshold for cls in animal_related_classes):
            buffer.append(frame_data)  # Buffer active segments
```

#### Mode 3: Animal Simulation Mode

- **Use Case:** Testing with pre-recorded animal sounds
- **Pipeline:** Identical to Recording Mode but with different input handling
- **Trigger:** Any other value for `audio_event['audioFile']`

### Weather Classification Pipeline

The engine includes experimental weather detection:

```python
def weather_pipeline(self, audio_clip):
    # Lower sample rate for weather (16kHz vs 48kHz for animals)
    audio, sample_rate = librosa.load(file, sr=self.config['WEATHER_SAMPLE_RATE'])
    
    # Fixed 2-second clips
    required_samples = self.config['WEATHER_SAMPLE_RATE'] * self.config['WEATHER_CLIP_DURATION']
    
    # Generate log-mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=16000, ...)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, top_db=80)
    
    # Resize to 260x260x3
    spectrogram_resized = tf.image.resize(log_mel_spectrogram[...], [260, 260])
    spectrogram_resized = np.repeat(spectrogram_resized, 3, axis=-1)
    
    return spectrogram_resized, audio, sample_rate
```

**Model Endpoint:** `http://ts-echo-model-cont:8501/v1/models/weather_model/versions/1:predict`

---

## Architecture & Components

### File Structure

```
src/Components/Engine/
├── echo_engine.py              # Main application entry point
├── echo_engine.json            # Configuration (audio params, endpoints)
├── echo_credentials.json       # Database credentials (gitignored)
├── echo_engine.sh              # Startup script with GCloud auth
├── requirements.txt            # Python dependencies
├── Engine.Dockerfile           # Container definition for engine
├── Model.Dockerfile            # Container definition for TF Serving
├── models.config               # TensorFlow Serving model configuration
├── README.md                   # Docker deployment instructions
├── class_names.json            # Species label mappings
├── test_equivalence            # TF/ONNX model comparison script
├── test_matches                # Prediction accuracy tests
├── helpers/
│   └── melspectrogram_to_cam.py   # CAM visualization utilities
├── yamnet_dir/                 # YAMNet model and metadata
│   ├── model/                  # TF SavedModel format
│   ├── yamnet.h5               # Pre-trained weights
│   ├── yamnet_class_map.csv    # AudioSet class mappings
│   ├── class_names.pkl         # Serialized species names
│   ├── label_encoder.pkl       # Sklearn label encoder
│   ├── params.py               # YAMNet configuration
│   └── yamnet.py               # YAMNet model definition
└── models/                     # Versioned model storage
    ├── echo_model/
    │   └── 1/                  # TF SavedModel v1
    │       └── echo_model.onnx # ONNX export (optional)
    └── weather_model/
        └── 1/                  # Weather classification model
```

### Key Components

| Component | Location | Responsibility | Input | Output |
|-----------|----------|----------------|-------|--------|
| EchoEngine | `echo_engine.py` (class) | Main orchestration, MQTT handling | Audio events via MQTT | Detection events to MongoDB |
| `combined_pipeline()` | `echo_engine.py` | Audio → Mel spectrogram transformation | Binary audio data | (260, 260, 3) tensor |
| `sound_event_detection()` | `echo_engine.py` | YAMNet-based segmentation | Long audio files | DataFrame with timestamped detections |
| `weather_pipeline()` | `echo_engine.py` | Weather-specific preprocessing | Binary audio (2s clips) | (260, 260, 3) tensor |
| `predict_class()` | `echo_engine.py` | Argmax + confidence extraction | Model logits | Species name + probability |
| `melspectrogram_to_cam` | `helpers/` | Class Activation Mapping for interpretability | Spectrogram tensor | CAM overlay image |
| TF Serving | Docker container | Model hosting and inference | JSON with spectrogram | Classification logits |
| MongoDB | External | Persistent storage | Detection events | Queries for API/dashboard |

### Dependencies & Rationale

```python
# requirements.txt highlights
tensorflow==2.15.0           # Core DL framework (GPU-enabled)
librosa                      # Industry standard for audio feature extraction
paho-mqtt==1.6.1             # MQTT client for IoT integration
pymongo[srv]                 # MongoDB driver with DNS seedlist support
google-cloud-storage         # GCP bucket access for species metadata
soundfile                    # Audio I/O (libsndfile wrapper)
audiomentations              # Data augmentation (used in training)
geopy                        # Location uncertainty calculations
```

**Why These Libraries?**
- **TensorFlow 2.15.0**: Stable release with TensorFlow Serving compatibility
- **Librosa**: Provides optimized Mel spectrogram implementations
- **MQTT**: Lightweight protocol ideal for IoT sensor networks
- **MongoDB**: Document database suited for JSON-like detection events

---

## Configuration & Credentials

### echo_engine.json

Complete configuration reference:

```json
{
    "AUDIO_DATA_DIRECTORY": "d:\\data\\b3",
    "CACHE_DIRETORY": "d:\\pipeline_cache",
    
    // Audio Processing Parameters
    "AUDIO_CLIP_DURATION": 5,              // seconds
    "AUDIO_SAMPLE_RATE": 48000,            // Hz
    "AUDIO_NFFT": 2048,                    // FFT window size
    "AUDIO_WINDOW": 500,                   // STFT window length
    "AUDIO_STRIDE": 200,                   // Hop length between frames
    "AUDIO_MELS": 260,                     // Number of Mel bands
    "AUDIO_FMIN": 20,                      // Min frequency (Hz)
    "AUDIO_FMAX": 13000,                   // Max frequency (Hz)
    "AUDIO_TOP_DB": 80,                    // Dynamic range for dB conversion
    
    // Weather Detection Parameters
    "WEATHER_CLIP_DURATION": 2,            // seconds
    "WEATHER_SAMPLE_RATE": 16000,          // Hz (lower for weather)
    
    // Model Input Specifications
    "MODEL_INPUT_IMAGE_WIDTH": 260,
    "MODEL_INPUT_IMAGE_HEIGHT": 260,
    "MODEL_INPUT_IMAGE_CHANNELS": 3,
    
    // MQTT Configuration
    "MQTT_CLIENT_URL": "ts-mqtt-server-cont",
    "MQTT_CLIENT_PORT": 1883,
    "MQTT_PUBLISH_URL": "projectecho/engine/2",
    
    // Google Cloud Platform
    "GCLOUD_PROJECT": "sit-23t1-project-echo-25288b9",
    "BUCKET_NAME": "project_echo_bucket_1",
    
    // TensorFlow Serving Endpoints
    "MODEL_SERVER": "http://ts-echo-model-cont:8501/v1/models/echo_model/versions/1:predict",
    "WEATHER_SERVER": "http://ts-echo-model-cont:8501/v1/models/weather_model/versions/1:predict",
    
    // Database
    "DB_HOSTNAME": "ts-mongodb-cont"
}
```


### Google Cloud Authentication

The engine requires GCP credentials to load species names from Cloud Storage:

```bash
# Authenticate (creates ~/.config/gcloud/application_default_credentials.json)
gcloud auth application-default login
```

This is handled automatically by `echo_engine.sh` in Docker deployments.

---

## Deployment (Docker)

### Architecture

The production deployment uses four interconnected containers:

1. **ts-mqtt-server-cont**: MQTT broker (Eclipse Mosquitto)
2. **ts-echo-model-cont**: TensorFlow Serving (hosts echo_model + weather_model)
3. **ts-echo-engine-cont**: Echo Engine (MQTT subscriber + inference client)
4. **ts-mongodb-cont**: MongoDB database

### Setup Steps

#### 1. Create Docker Network

```bash
docker network create --driver bridge echo-net
```

#### 2. Create Persistent Volume

```bash
docker volume create myvolume
```

This stores GCP credentials across container restarts.

#### 3. Build Model Server

```bash
docker build --file Model.Dockerfile --build-arg BASE_IMAGE=tensorflow/serving:2.15.0 -t ts-echo-model .
```

> **Critical:** Ensure trained models exist in `models/echo_model/1/` before building!

```bash
docker run -p 8501:8501 --name ts-echo-model-cont --network echo-net -d ts-echo-model
```

**Verify Models Are Loaded:**

```bash
curl http://localhost:8501/v1/models/echo_model/metadata
curl http://localhost:8501/v1/models/weather_model/metadata
```

#### 4. Build & Run Echo Engine

```bash
docker build --file Engine.Dockerfile --build-arg BASE_IMAGE=tensorflow/tensorflow:2.15.0 -t ts-echo-engine .

docker run --name ts-echo-engine-cont -it --rm \
  -v myvolume:/root \
  --network echo-net \
  ts-echo-engine
```

**On First Run:** The container will prompt for GCloud authentication. Follow the URL to authorize, then paste the token.

### models.config

TensorFlow Serving multi-model configuration:

```
model_config_list: {
  config: [
    {
      name: "echo_model",
      base_path: "/models/echo_model",
      model_platform: "tensorflow",
      model_version_policy: {all{}}  // Serve all versions
    },
    {
      name: "weather_model",
      base_path: "/models/weather_model",
      model_platform: "tensorflow",
      model_version_policy: {all{}}
    }
  ]
}
```

### Container Health Checks

```bash
# Check engine logs
docker logs ts-echo-engine-cont

# Check model server status
docker exec ts-echo-model-cont curl localhost:8501/v1/models/echo_model

# Test MQTT connection
docker exec ts-echo-engine-cont nc -zv ts-mqtt-server-cont 1883
```

---

## Edge Cases & Warnings

### Audio Duration Mismatch

**Issue:** Input files shorter than `AUDIO_CLIP_DURATION` (default: 5 seconds) may cause dimension errors or produce low-confidence predictions due to zero-padding.

**Handling:** The `load_random_subsection()` method automatically zero-pads short clips:

```python
if audio_duration_secs <= duration_secs:
    padding_length = duration_secs * AUDIO_SAMPLE_RATE - len(audio)
    padding = tf.zeros([padding_length], dtype=audio.dtype)
    subsection = tf.concat([audio, padding], axis=0)
```

**Impact:** Zero-padded regions produce silent spectrograms, which can:
- Lower confidence scores for legitimate detections
- Increase false negatives for short vocalizations

**Mitigation:** For datasets with < 5s clips, consider:
- Reducing `AUDIO_CLIP_DURATION` in config
- Retraining models on shorter windows
- Using `load_specific_subsection()` for precise timing

### FFmpeg Binary Dependency

**Gotcha:** Python's ffmpeg packages are wrappers only. The actual binary must be in your system `$PATH`.

**Symptoms of Missing FFmpeg:**

```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

**Verification:**

```bash
which ffmpeg        # Should return /usr/bin/ffmpeg or similar
ffmpeg -version     # Should display version info
```

**Docker Note:** The `Engine.Dockerfile` installs system packages via `apt-get`, but ensure base images include required libraries.

### Sample Rate Drifting

**Critical Warning:** Training and inference must use the same sample rate.

| Scenario | Result |
|----------|--------|
| Train: 48kHz, Infer: 48kHz |  Correct |
| Train: 48kHz, Infer: 16kHz |  Spectrograms misaligned, predictions invalid |
| Train: 16kHz, Infer: 48kHz |  Frequency bins don't match model expectations |

**Root Cause:** Mel filterbanks are calculated based on sample rate. Changing SR shifts the frequency-to-bin mapping.

**Prevention:**
1. Always verify `config.json` matches recording hardware SR
2. Lock `AUDIO_SAMPLE_RATE` during training
3. Document model metadata with expected SR

### MQTT Connection Retry Loop

The engine retries MQTT connection indefinitely:

```python
connected = False
while not connected:
    try:
        client.connect(self.config['MQTT_CLIENT_URL'], self.config['MQTT_CLIENT_PORT'])
        connected = True
    except:
        time.sleep(1)  # Retry every second
```

**Implication:** If MQTT broker is down, the engine will hang. Monitor logs for:

```
Subscribing to MQTT: ts-mqtt-server-cont projectecho/engine/2
```

If this line never appears, check broker status.

### Model Version Mismatch

**Issue:** TensorFlow Serving expects models in versioned directories (`models/echo_model/1/`). Incorrect paths cause startup failures.

**Symptoms:**

```
E tensorflow_serving/util/file_probing.cc: No versions of servable echo_model found
```

**Fix:** Ensure directory structure matches `models.config`:

```
models/
└── echo_model/
    └── 1/              # Version number directory
        ├── saved_model.pb
        └── variables/
```

### GCP Authentication in Containers

**Issue:** The `gcp_load_species_list()` method requires valid GCP credentials. Without them:

```
google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials
```

**Solution:** The `echo_engine.sh` script handles this:

```bash
if [ ! -f ~/.config/gcloud/application_default_credentials.json ]; then
    gcloud auth application-default login
fi
```

**Docker Volume:** Mount credentials via volume:

```bash
-v myvolume:/root
```

This persists credentials across container restarts.

### Memory Leaks in Long-Running Processes

**Observation:** The `client.loop_forever()` call runs indefinitely. Memory profiling shows gradual accumulation in numpy arrays.

**Mitigation:** Implement periodic garbage collection:

```python
import gc
# In on_message(), after every N messages:
if message_count % 100 == 0:
    gc.collect()
```

### Coordinate Randomization for Privacy

In `Recording_Mode_V2`, multiple detections from the same audio clip get randomized locations:

```python
new_lat, new_lon = self.generate_random_location(lat, lon, 50, 100)
```

**Purpose:** Prevents exact sensor triangulation while maintaining ~50-100m geolocation accuracy.

**Warning:** This is a privacy feature, not a bug. Do not remove without understanding implications.

---

# Project Echo - Prototypes Documentation

## Table of Contents

1. Overview
2. Torch Implementation (Audio Classification)
3. Computer Vision (Image Classification)
4. Integration Guide

---

## Overview

The Prototypes directory contains experimental implementations for future deployment:

| Prototype | Location | Purpose | Status |
|-----------|----------|---------|--------|
| **Torch Implementation** | `src/Prototypes/engine/torch_impl` | PyTorch-based audio classification for edge/IoT deployment | Active Development |
| **Computer Vision** | `src/Prototypes/Computer Vision` | Image-based creature classification using camera traps | Prototype |

---

## Torch Implementation (Audio Classification)

### Purpose

The torch_impl is a PyTorch-based training and inference pipeline designed for:
- Training deep learning models for wildlife audio classification
- Converting models to lightweight formats (ONNX, TFLite) for edge deployment
- Benchmarking inference performance across different backends

### File Structure

```
src/Prototypes/engine/torch_impl/
├── config/                         # Training configuration files
├── docs/                           # Documentation
├── model/                          # Model architecture definitions
├── model_server/                   # Multi-format inference server
│
├── train.py                        # Main training script
├── dataset.py                      # Data loading and preprocessing
├── augment.py                      # Audio augmentation utilities
├── main.py                         # Entry point
├── _single.py                      # Single file inference
│
├── pytorch_to_onnx.py              # PyTorch → ONNX conversion
├── convert_onnx_to_tflite.py       # ONNX → TFLite conversion
├── validate_models.py              # Model equivalence testing
├── operator_test.py                # TFLite operator compatibility
│
├── benchmark_server.py             # Latency/throughput benchmarking
├── test_audio_preprocess.py        # Preprocessing comparison (librosa vs torchaudio)
├── validating_audio_file.py        # Audio file validation
├── data_analysis.py                # Dataset analysis utilities
│
├── light_echo_engine.py            # Lightweight inference engine
├── light_echo_engine.json          # Engine configuration
├── light_echo_credentials.json     # Credentials template
├── light_engine.Dockerfile         # Docker deployment
├── deploy_model.py                 # Model deployment utilities
├── demo_iot.py                     # IoT demonstration script
│
├── filter_noise.sh                 # Background noise filtering (ESC-50)
├── tmp.py                          # Quantization demo
├── b3.joblib                       # Cached dataset
├── b3_data_analysis.png            # Dataset visualization
│
├── Knowledge_Distillation_With_QAT.ipynb  # Knowledge distillation notebook
├── single_file_pipeline_colab.ipynb       # Colab-compatible pipeline
│
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project configuration
├── uv.lock                         # Dependency lock file
├── .python-version                 # Python version (3.11)
└── README.md                       # Documentation
```

---

### Training Pipeline

#### 1. Environment Setup

```bash
cd src/Prototypes/engine/torch_impl

# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

**Python Version:** 3.11 (specified in `.python-version`)

#### 2. Configuration

Training parameters are stored in `config/` directory. Example configuration:

```yaml
# config/efficientnet_v2_qat.yaml
model:
  name: efficientnetv2_s
  num_classes: 118
  pretrained: true

audio:
  sample_rate: 48000
  clip_duration: 5
  n_fft: 1024
  hop_length: 320
  n_mels: 64
  fmin: 50
  fmax: 14000
  top_db: 80

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  
loss:
  type: circle_loss
  scale: 80.0
  margin: 0.4
```

#### 3. Dataset Preparation

Audio files should be organized by species:

```
data/
├── Alectura_lathami/
│   ├── recording_001.wav
│   ├── recording_002.wav
│   └── ...
├── Capra_hircus/
│   ├── recording_001.wav
│   └── ...
└── [118 species folders]
```

**Analyze your dataset:**

```bash
python data_analysis.py
```

This generates `b3_data_analysis.png` showing class distribution.

#### 4. Training

```bash
python train.py --config config/efficientnet_v2_qat.yaml
```

**Training Features:**
- EfficientNetV2-S backbone with ArcFace head
- CircleLoss for metric learning (s=80, m=0.4)
- Mixed precision training (AMP)
- Quantization Aware Training (QAT) support
- UMAP visualization of embeddings

**Output:**
- Model checkpoints: `checkpoints/`
- Training logs: `logs/`
- UMAP visualizations: `umap_*.png`

#### 5. Model Architecture

The model uses EfficientNetV2 with a cosine similarity head:

```python
# model/effv2.py
class EfficientNetV2ArcFace(nn.Module):
    def __init__(self, num_classes=118, embedding_dim=512):
        self.backbone = efficientnet_v2_s(pretrained=True)
        self.backbone.classifier = nn.Identity()
        self.embedding = nn.Linear(1280, embedding_dim)
        self.head = CosineLinear(embedding_dim, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        cosine_sim = self.head(embeddings)  # Range: [-1, 1]
        return cosine_sim
```

**CosineLinear Head:**

```python
# model/utils.py
class CosineLinear(nn.Module):
    def forward(self, input):
        # Normalize both input and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine  # Raw cosine similarities
```

**CircleLoss:**

```python
# model/utils.py
class CircleLoss(nn.Module):
    def __init__(self, s=80.0, m=0.4):
        self.s = s  # Scale factor
        self.m = m  # Margin
    
    def forward(self, cosine, label):
        # s_p = s * (cosine - (1 - m)) for positive class
        # s_n = s * (cosine - m) for negative classes
        ...
```

---

### Model Conversion Pipeline

#### Step 1: PyTorch → ONNX

```bash
python pytorch_to_onnx.py
```

**Deliverable:** `pytorch_to_onnx.py`

#### Step 2: ONNX → TFLite

```bash
python convert_onnx_to_tflite.py
```

**Deliverable:** `convert_onnx_to_tflite.py`

#### Step 3: Validate Conversion

```bash
python validate_models.py 
```

**Deliverable:** `validate_models.py`

#### Step 4: Check Operator Compatibility

```bash
python operator_test.py --model model.tflite
```

Verifies all operators are supported by standard TFLite runtime.

**Deliverable:** `operator_test.py`

---

### Inference & Deployment

#### Light Echo Engine

Lightweight inference engine for edge deployment:

```bash
python light_echo_engine.py
```

**Configuration:** `light_echo_engine.json`

```json
{
    "AUDIO_SAMPLE_RATE": 48000,
    "AUDIO_CLIP_DURATION": 5,
    "AUDIO_NFFT": 1024,
    "AUDIO_STRIDE": 320,
    "AUDIO_MELS": 64,
    "MODEL_SERVER": "http://model-server:8501/v1/models/echo_model/versions/1:predict"
}
```

#### Model Server (Multi-Format)

Supports TFLite, ONNX, and PyTorch backends:

```bash
cd model_server

# Start with TFLite
MODEL_PATH=model.tflite python server.py

# Start with ONNX
MODEL_PATH=model.onnx python server.py
```

**Files:**
- `model_server/inference_engine.py` - Multi-backend engine
- `model_server/server.py` - Flask API server
- `model_server/Model.Dockerfile` - Docker deployment

#### Benchmarking

```bash
python benchmark_server.py \
    --url http://localhost:8501 \
    --backend tflite \
    --requests 100
```

**Deliverable:** `benchmark_server.py`

---

### Key Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `train.py` | Train model | `python train.py --config config/xxx.yaml` |
| `dataset.py` | Data loading | Imported by train.py |
| `augment.py` | Audio augmentation | Imported by dataset.py |
| `pytorch_to_onnx.py` | Export to ONNX | `python pytorch_to_onnx.py --checkpoint model.pt` |
| `convert_onnx_to_tflite.py` | Convert to TFLite | `python convert_onnx_to_tflite.py --input model.onnx` |
| `validate_models.py` | Verify conversion | `python validate_models.py --pytorch x --tflite y` |
| `operator_test.py` | Check TFLite ops | `python operator_test.py --model model.tflite` |
| `benchmark_server.py` | Performance testing | `python benchmark_server.py --url http://...` |
| `test_audio_preprocess.py` | Compare preprocessing | `python test_audio_preprocess.py` |
| `light_echo_engine.py` | Edge inference | `python light_echo_engine.py` |
| `demo_iot.py` | IoT demonstration | `python demo_iot.py` |


## Computer Vision (Image Classification)

### Purpose

Camera trap image classification for wildlife monitoring. Uses MobileNetV2 for edge deployment.

### File Structure

```
src/Prototypes/Computer Vision/
├── Create Dataset/
│   ├── get_creatures_v1.py              # Download images from Wikimedia
│   ├── augment_creatures_v1.py          # Data augmentation
│   └── How to Create A Creatures Dataset_v1.docx
│
├── Train Model/
│   ├── train_model_v1.py                # Training script
│   ├── requirements.txt
│   └── Computer Vision on Edge Devices_v1.docx
│
├── Inference/
│   ├── classify_stills_v1.py            # Image classification
│   ├── classify_video_v1.py             # Video classification
│   └── requirements.txt
│
└── readme.md
```

---

### Pipeline Overview

```
Step 1: Create Dataset
    get_creatures_v1.py → augment_creatures_v1.py
                ↓
Step 2: Train Model
    train_model_v1.py → SavedModel + TFLite
                ↓
Step 3: Inference
    classify_stills_v1.py / classify_video_v1.py
```

---

### Step 1: Create Dataset

#### Download Images

```bash
cd "Create Dataset"
python get_creatures_v1.py
```

**What it does:**
- Downloads images from Wikimedia Commons
- Converts to 300×300 RGB
- Outputs ZIP file + folder

#### Augment Dataset

```bash
python augment_creatures_v1.py
```

**Augmentations applied:**
- Horizontal/vertical flips
- Rotation
- Gaussian blur
- Random noise
- Color shifts

**Output:** Thousands of augmented images per class

---

### Step 2: Train Model

```bash
cd "Train Model"
pip install -r requirements.txt
python train_model_v1.py
```

**Model:** MobileNetV2 (transfer learning)

**Outputs:**
- TensorFlow SavedModel
- Quantized TFLite model
- Accuracy/loss plots

**Training Time:** ~50 minutes for 5 classes × 1,000 images (i7 CPU, 32GB RAM)

---

### Step 3: Inference

#### Classify Still Images

```bash
cd Inference
pip install -r requirements.txt
python classify_stills_v1.py
```

#### Classify Video

```bash
python classify_video_v1.py
```

Press `Q` to quit the window.

**Configuration:**
- Update `MODEL_PATH` to point to your TFLite model
- Class order must match training folder order

---

### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Wrong predictions | Class order mismatch | Ensure inference classes match training folder order |
| Video slow | Processing every frame | Increase `FRAME_STRIDE` |
| "No images found" | Incorrect paths | Check folder paths in script |

---

## Integration Guide

### How Prototypes Relate to Production

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRODUCTION (src/Components/Engine)           │
│                                                                 │
│   IoT Sensors → MQTT → Echo Engine → TF Serving → MongoDB       │
│                              ↑                                  │
│                              │ Models                           │
└──────────────────────────────┼──────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────┐
│                     PROTOTYPES                                   │
│                              │                                  │
│   ┌──────────────────────────┴───────────────────────────┐      │
│   │                                                      │      │
│   │  torch_impl/              Computer Vision/           │      │
│   │  ├── train.py             ├── train_model_v1.py     │      │
│   │  ├── pytorch_to_onnx.py   ├── classify_video_v1.py  │      │
│   │  └── model.tflite         └── model.tflite          │      │
│   │         │                         │                  │      │
│   │         ▼                         ▼                  │      │
│   │    Audio Models              Image Models            │      │
│   │                                                      │      │
│   └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Workflow

1. **Train in Prototypes** → Export TFLite model
2. **Test with Light Engine** → Validate accuracy
3. **Benchmark** → Measure latency/throughput
4. **Deploy to Production** → Copy to `src/Components/Engine/models/`

---

## Quick Reference

### Audio Classification (torch_impl)

```bash
# Train
python train.py --config config/efficientnet_v2_qat.yaml

# Convert
python pytorch_to_onnx.py --checkpoint model.pt --output model.onnx
python convert_onnx_to_tflite.py --input model.onnx --output model.tflite

# Validate
python validate_models.py --pytorch model.pt --tflite model.tflite

# Benchmark
python benchmark_server.py --url http://localhost:8501 --requests 100

# Deploy
python light_echo_engine.py
```

