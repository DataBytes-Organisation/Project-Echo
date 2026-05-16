"""
Manual integration test: publishes a test IoT MQTT message.

Start the engine stack first (docker-compose up), then run this script
to send a fake audio detection and verify the handler processes it.

Uses only stdlib + paho-mqtt + numpy (no TF/librosa needed).

Usage:
    pip install paho-mqtt numpy
    python test_iot_publisher.py
"""

import json
import base64
import wave
import io
import struct
import argparse

import numpy as np
import paho.mqtt.client as mqtt


def generate_test_wav(duration_secs=5, sample_rate=48000, freq_hz=440):
    """Generate a sine-wave WAV file as base64, using only stdlib + numpy."""
    t = np.linspace(0, duration_secs, int(duration_secs * sample_rate), dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * freq_hz * t) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Publish a test IoT MQTT message")
    parser.add_argument("--broker", default="broker.hivemq.com", help="MQTT broker address")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--topic", default="iot/data/test", help="MQTT topic")
    parser.add_argument("--lat", type=float, default=-37.8136, help="GPS latitude")
    parser.add_argument("--lon", type=float, default=144.9631, help="GPS longitude")
    parser.add_argument("--sensor-id", default="test_rpi_01", help="Sensor identifier")
    args = parser.parse_args()

    payload = {
        "audio_file": generate_test_wav(),
        "gps_data": {
            "lat": args.lat,
            "lon": args.lon,
        },
        "sensor_id": args.sensor_id,
        "gps_uncertainty": 15.0,
    }

    client = mqtt.Client()
    client.connect(args.broker, args.port, 60)
    client.publish(args.topic, json.dumps(payload))
    print(f"Published test message to {args.broker}:{args.port}/{args.topic}")
    print(f"  sensor_id : {args.sensor_id}")
    print(f"  lat/lon   : {args.lat}, {args.lon}")
    print(f"  audio     : {len(payload['audio_file'])} chars base64")
    client.disconnect()


if __name__ == "__main__":
    main()
