import os
import sounddevice as sd
import numpy as np
import paho.mqtt.client as mqtt
from gps3 import gps3
import time
import wave
import psutil
import json
import base64

AUDIO_DIR = "audioLocal"
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "iot/data/test"

os.makedirs(AUDIO_DIR, exist_ok=True)

sd.default.device = (1, None)

print("Connecting to MQTT Broker...")
client = mqtt.Client()
client.connect(BROKER, PORT, 60)
client.loop_start()
topic = TOPIC
print("Connected.")

def on_publish(client, userdata, mid):
    print(f"Message {mid} published.")

def record_audio(duration=5, samplerate=44100):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"audio_{timestamp}.wav"
    filepath = os.path.join(AUDIO_DIR, filename)

    print("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    print("Recording complete", filename)
    return filepath


def get_health_report():
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
        "uptime": time.time() - psutil.boot_time()
    }

def get_gps_reading():
    gps_socket = gps3.GPSDSocket()
    data_stream = gps3.DataStream()

    gps_socket.connect()
    gps_socket.watch()

    for new_data in gps_socket:
        if new_data:
            data_stream.unpack(new_data)

            lat = data_stream.TPV.get('lat')
            lon = data_stream.TPV.get('lon')
            mode = data_stream.TPV.get('mode')

            if mode == 3 and lat != 'n/a' and lon != 'n/a':
                print("GPS FIX:", lat, lon)
                return {"lat": lat, "lon": lon}

    return None

def send_data():
    audio_file = record_audio()
    
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    
    health_data = get_health_report()

    gps_data = get_gps_reading()

    audio_b64 = base64.b64encode(audio_bytes).decode('ascii')

    payload = {
        "health_data": health_data,
        "gps_data": gps_data,
        "audio_file": audio_b64
    }

    json_payload = json.dumps(payload)

    client.on_publish = on_publish
    client.publish(topic, json_payload, qos=1)
    print(f"Published json + audio to {topic}: {audio_file}")

if __name__ == "__main__":
    while True:
        send_data()
        time.sleep(20)
