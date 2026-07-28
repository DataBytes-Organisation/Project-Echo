# IoT Edge Inference — EfficientNetV2 TFLite on Raspberry Pi

## Overview

This document covers all changes made to support **on-device ML inference** for Project Echo IoT nodes, and explains how to deploy and run the edge client.

Previously, IoT devices sent raw audio over MQTT to the engine server, which ran the model. Now a Raspberry Pi can run the EfficientNetV2 TFLite model locally and publish only the prediction result — no audio is transmitted over the network.

---

## Architecture

### Before (server-side inference)
```
RPi → [raw audio bytes, base64, ~500KB/clip] → MQTT → Engine → TF Serving → API → MongoDB
```

### After (edge inference — this PR)
```
RPi → [species + confidence + GPS, ~300 bytes] → MQTT → Engine → API → MongoDB
```

Both modes are supported simultaneously on the same MQTT topic. The engine detects which path to use based on the payload.

---

## Files Changed

### New files

| File | Purpose |
|---|---|
| `src/Components/IoT/edge_inference/iot_edge_client.py` | RPi script — records audio, runs TFLite, publishes prediction |
| `src/Components/IoT/edge_inference/requirements.txt` | RPi Python dependencies |
| `src/Components/IoT/edge_inference/README.md` | This file |

### Modified files

| File | Change |
|---|---|
| `src/Components/Engine/echo_engine_iot.py` | Added `_handle_edge_prediction()` method; `on_iot_message()` now routes on `payload["type"]` |
| `src/Components/Engine/echo_engine_iot.py` | Removed duplicate `import base64` |
| `src/Components/Engine/echo_engine_iot.py` | Hardcoded API URL replaced with `self.config['API_URL']` |
| `src/Components/Engine/echo_engine.json` | Added `"API_URL"` field |
| `src/Components/Engine/Engine.Dockerfile` | Now copies `echo_engine_iot.py` as `echo_engine.py` so IoT engine runs in Docker |
| `src/Components/Engine/requirements.txt` | Added `scikit-learn` (was imported but missing) |
| `src/Components/Engine/test_iot_integration.py` | Updated import to `from echo_engine_iot import EchoEngine` so tests run locally |
| `src/Prototypes/engine/torch_impl/light_echo_engine.json` | Fixed `MQTT_CLIENT_URL` from `"mqtt-broker"` → `"ts-mqtt-server-cont"` (was causing DNS failure) |
| `src/Prototypes/engine/torch_impl/requirements.txt` | Added `paho-mqtt==1.6.1`, `pymongo`, `geopy`, `google-cloud-storage` |

---

## Engine: How the Two Modes Work

The engine's `on_iot_message()` handler now routes based on the payload `type` field:

```
MQTT message received
        │
        ▼
payload["type"] == "prediction"?
     Yes ──► _handle_edge_prediction()  ──► POST /engine/event  (no ML on server)
     No  ──► audio_file present?
                  Yes ──► combined_pipeline() → TF Serving → POST /engine/event
                  No  ──► drop message
```

**Edge prediction payload** (sent by `iot_edge_client.py`):
```json
{
    "type": "prediction",
    "timestamp": "1746700800",
    "sensor_id": "rpi_edge_node_1",
    "species": "Acanthiza chrysorrhoa",
    "confidence": 93.37,
    "top5": [
        {"label": "Acanthiza chrysorrhoa", "confidence": 93.37},
        {"label": "Malurus cyaneus",       "confidence":  3.12}
    ],
    "gps_data": {"lat": -37.8136, "lon": 144.9631},
    "gps_uncertainty": 10.0
}
```

**Raw audio payload** (sent by the original `client.py` — still supported):
```json
{
    "audio_file": "<base64 WAV>",
    "gps_data": {"lat": -37.8136, "lon": 144.9631},
    "sensor_id": "rpi_node_1"
}
```

---

## Deploying the Edge Client on a Raspberry Pi

### 1. Copy model files to the RPi

You need three files from `src/Prototypes/engine/torch_impl/Integrate_EfficientNetV2_Engine/_trained_models/`:

```
efficientnetv2_project_echo.tflite
class_mapping.json
preprocess_config.json
```

Create a `models/` directory next to `iot_edge_client.py` and place them there:

```
edge_inference/
├── iot_edge_client.py
├── requirements.txt
├── models/
│   ├── efficientnetv2_project_echo.tflite
│   ├── class_mapping.json
│   └── preprocess_config.json
```

### 2. Install dependencies

On the Raspberry Pi (Python 3.9+):

```bash
pip install -r requirements.txt
```

> `tflite-runtime` is the lightweight TFLite package (~1 MB) designed for embedded devices.
> If you have full TensorFlow installed, the script falls back to `tensorflow.lite` automatically.

### 3. Run with fixed GPS coordinates

Use this during development or when no GPS module is attached:

```bash
python iot_edge_client.py \
    --broker broker.hivemq.com \
    --port 1883 \
    --topic iot/data/test \
    --sensor-id rpi_edge_node_1 \
    --lat -37.8136 \
    --lon 144.9631 \
    --interval 10
```

### 4. Run with live GPS (gpsd)

If a GPS module is connected and `gpsd` is running:

```bash
python iot_edge_client.py \
    --sensor-id rpi_edge_node_1 \
    --use-gps \
    --interval 10
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--broker` | `broker.hivemq.com` | MQTT broker hostname |
| `--port` | `1883` | MQTT broker port |
| `--topic` | `iot/data/test` | MQTT topic to publish on |
| `--sensor-id` | `rpi_edge_node_1` | Unique identifier for this RPi node |
| `--lat` | `-37.8136` | Fixed latitude (used when `--use-gps` not set) |
| `--lon` | `144.9631` | Fixed longitude (used when `--use-gps` not set) |
| `--gps-uncertainty` | `10.0` | GPS accuracy estimate in metres |
| `--interval` | `10.0` | Seconds between recordings |
| `--use-gps` | off | Read GPS from `gpsd` instead of fixed coordinates |

---

## Running Unit Tests

Unit tests cover payload validation, happy-path handler, config keys, and startup order. They mock all heavy dependencies so no Docker stack or GPU is needed.

```bash
cd src/Components/Engine
python -m pytest test_iot_integration.py -v
```

Expected output:
```
test_iot_integration.py::TestIoTPayloadValidation::test_empty_audio_file_rejected   PASSED
test_iot_integration.py::TestIoTPayloadValidation::test_missing_audio_file_rejected PASSED
test_iot_integration.py::TestIoTPayloadValidation::test_missing_gps_data_rejected   PASSED
test_iot_integration.py::TestIoTPayloadValidation::test_missing_lat_rejected        PASSED
test_iot_integration.py::TestIoTPayloadValidation::test_missing_lon_rejected        PASSED
test_iot_integration.py::TestIoTMessageHandler::...                                 PASSED
...
12 passed
```

## End-to-End Test (no RPi needed)

Publish a synthetic test event from any machine to verify the engine picks it up:

```bash
cd src/Components/Engine
python test_iot_publisher.py
```

This sends a 440 Hz WAV clip to `broker.hivemq.com` on `iot/data/test`. If the engine container is running, it will log:
```
Received IoT message...
IoT Predicted class : <species>
IoT Predicted probability : <value>
```

---

## Notes

- The MQTT broker (`broker.hivemq.com`) is a public test broker. For production, configure a private broker and set `IOT_MQTT_BROKER` in `echo_engine.json`.
- The model expects audio at **32 kHz** — the recording sample rate in `iot_edge_client.py` is set automatically from `preprocess_config.json`.
- On RPi 4 / RPi 5, TFLite inference on a 5-second clip takes approximately 200–800 ms.
