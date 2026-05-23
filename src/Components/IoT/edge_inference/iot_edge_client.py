#!/usr/bin/env python3
"""
IoT Edge Inference Client

Records audio from the microphone, classifies it on-device using the
EfficientNetV2 TFLite model, and publishes only the prediction result
(species + confidence + GPS) over MQTT — no audio is transmitted.

Model files expected in ./models/:
    efficientnetv2_project_echo.tflite
    class_mapping.json
    preprocess_config.json

Usage:
    python iot_edge_client.py [--broker BROKER] [--port PORT] [--topic TOPIC]
                              [--sensor-id ID] [--lat LAT] [--lon LON]
                              [--interval SECS] [--use-gps]
"""

import argparse
import json
import time
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import librosa
import paho.mqtt.client as mqtt

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite  # fallback if full TF installed

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
_DIR = Path(__file__).parent
MODEL_PATH        = _DIR / "models" / "efficientnetv2_project_echo.tflite"
CLASS_MAPPING_PATH = _DIR / "models" / "class_mapping.json"
PREPROCESS_CFG_PATH = _DIR / "models" / "preprocess_config.json"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    with open(PREPROCESS_CFG_PATH) as f:
        cfg = json.load(f)
    with open(CLASS_MAPPING_PATH) as f:
        index_to_label = json.load(f)["index_to_label"]

    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"TFLite model loaded. Input shape: {input_details[0]['shape']}", flush=True)
    return interpreter, input_details, output_details, cfg, index_to_label


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------
def record_audio(duration_s: float, sample_rate: int) -> np.ndarray:
    print(f"Recording {duration_s}s at {sample_rate} Hz...", flush=True)
    audio = sd.rec(
        int(duration_s * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


# ---------------------------------------------------------------------------
# Preprocessing  (matches efficientnetv2_preprocess_audio_bytes in the engine)
# ---------------------------------------------------------------------------
def preprocess(audio: np.ndarray, cfg: dict) -> np.ndarray:
    target_sr  = cfg["target_sr"]
    duration_s = cfg["duration_s"]
    n_mels     = cfg["n_mels"]
    hop_length = cfg["hop_length"]
    fmin       = cfg["fmin"]
    fmax       = cfg["fmax"]

    target_len = int(target_sr * duration_s)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        audio = audio[:target_len]

    mel    = librosa.feature.melspectrogram(
        y=audio, sr=target_sr, n_mels=n_mels,
        hop_length=hop_length, fmin=fmin, fmax=fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    # NCHW: [1, 1, n_mels, time_frames]
    return mel_db[np.newaxis, np.newaxis, :, :].astype(np.float32)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    interpreter, input_details, output_details,
    x: np.ndarray, index_to_label: dict,
) -> tuple[str, float, list]:
    expected_shape = tuple(input_details[0]["shape"])

    if tuple(x.shape) != expected_shape:
        # Try NHWC layout if model was exported that way
        x_nhwc = np.transpose(x, (0, 2, 3, 1))
        if tuple(x_nhwc.shape) == expected_shape:
            x = x_nhwc
        else:
            raise ValueError(
                f"Shape mismatch: model expects {expected_shape}, "
                f"got NCHW {x.shape} or NHWC {x_nhwc.shape}"
            )

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]["index"])[0].astype(np.float32)

    exp_vals = np.exp(logits - np.max(logits))
    probs    = exp_vals / np.sum(exp_vals)

    top_indices      = np.argsort(probs)[::-1][:5]
    predicted_index  = int(top_indices[0])
    confidence       = round(float(probs[predicted_index]) * 100.0, 2)
    species          = index_to_label[str(predicted_index)]

    top5 = [
        {
            "label":      index_to_label[str(int(i))],
            "confidence": round(float(probs[i]) * 100.0, 2),
        }
        for i in top_indices
    ]
    return species, confidence, top5


# ---------------------------------------------------------------------------
# GPS (optional — uses gpsd if available, otherwise falls back to fixed coords)
# ---------------------------------------------------------------------------
def get_gps(fallback_lat: float, fallback_lon: float) -> dict:
    try:
        from gps3 import gps3
        gps_socket  = gps3.GPSDSocket()
        data_stream = gps3.DataStream()
        gps_socket.connect()
        gps_socket.watch()
        for new_data in gps_socket:
            if new_data:
                data_stream.unpack(new_data)
                lat  = data_stream.TPV.get("lat")
                lon  = data_stream.TPV.get("lon")
                mode = data_stream.TPV.get("mode")
                if mode == 3 and lat != "n/a" and lon != "n/a":
                    return {"lat": float(lat), "lon": float(lon)}
    except Exception:
        pass
    return {"lat": fallback_lat, "lon": fallback_lon}


# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------
def connect_mqtt(broker: str, port: int) -> mqtt.Client:
    client    = mqtt.Client()
    connected = threading.Event()

    def on_connect(c, userdata, flags, rc):
        if rc == 0:
            print(f"MQTT connected to {broker}:{port}", flush=True)
            connected.set()
        else:
            print(f"MQTT connection failed (rc={rc})", flush=True)

    client.on_connect = on_connect
    client.connect(broker, port, keepalive=60)
    client.loop_start()

    if not connected.wait(timeout=10):
        raise RuntimeError(f"Could not connect to MQTT broker {broker}:{port}")
    return client


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------
def build_payload(
    species: str, confidence: float, top5: list,
    sensor_id: str, gps: dict, gps_uncertainty: float,
) -> dict:
    return {
        "type":            "prediction",
        "timestamp":       str(int(time.time())),
        "sensor_id":       sensor_id,
        "species":         species,
        "confidence":      confidence,
        "top5":            top5,
        "gps_data":        gps,
        "gps_uncertainty": gps_uncertainty,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="IoT edge inference client")
    parser.add_argument("--broker",          default="broker.hivemq.com")
    parser.add_argument("--port",            type=int,   default=1883)
    parser.add_argument("--topic",           default="iot/data/test")
    parser.add_argument("--sensor-id",       default="rpi_edge_node_1")
    parser.add_argument("--lat",             type=float, default=-37.8136)
    parser.add_argument("--lon",             type=float, default=144.9631)
    parser.add_argument("--gps-uncertainty", type=float, default=10.0)
    parser.add_argument("--interval",        type=float, default=10.0,
                        help="Seconds between recordings (default: 10)")
    parser.add_argument("--use-gps",         action="store_true",
                        help="Read GPS from gpsd instead of using fixed --lat/--lon")
    args = parser.parse_args()

    interpreter, input_details, output_details, cfg, index_to_label = load_model()
    client = connect_mqtt(args.broker, args.port)

    print("Edge inference client running. Ctrl+C to stop.\n", flush=True)
    try:
        while True:
            gps = get_gps(args.lat, args.lon) if args.use_gps else {"lat": args.lat, "lon": args.lon}

            audio   = record_audio(cfg["duration_s"], cfg["target_sr"])
            x       = preprocess(audio, cfg)
            species, confidence, top5 = run_inference(
                interpreter, input_details, output_details, x, index_to_label
            )

            print(f"  → {species}  {confidence:.1f}%", flush=True)
            for entry in top5[1:]:
                print(f"     {entry['label']}: {entry['confidence']:.1f}%", flush=True)

            payload = build_payload(
                species, confidence, top5,
                args.sensor_id, gps, args.gps_uncertainty,
            )
            client.publish(args.topic, json.dumps(payload), qos=1)
            print(f"Published to {args.topic}\n", flush=True)

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("Stopping.", flush=True)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
