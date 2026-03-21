import os
import sounddevice as sd
import numpy as np
import paho.mqtt.client as mqtt
import time
import wave

SAVE_DIR = "audioLocal"
os.makedirs(SAVE_DIR, exist_ok=True)

sd.default.device = (1, None)

print("Connecting to MQTT Broker...")
client = mqtt.Client()
client.connect("broker.hivemq.com", 1883, 60)
client.loop_start()
topic = "iot/data/test"
print("Connected.")

def record_audio(duration=5, samplerate=44100):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"audio_{timestamp}.wav"
    filepath = os.path.join(SAVE_DIR, filename)

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

def on_publish(client, userdata, mid):
    print(f"Message {mid} published.")

def send_data():
    audio_file = record_audio()
    
    with open(audio_file, "rb") as f:
        imagestring = f.read()
    audio_data = bytearray(imagestring)


    client.on_publish = on_publish
    client.publish(topic, audio_data, qos=1)
    print(f"Published audio to {topic}: {audio_file}")

if __name__ == "__main__":
    while True:
        send_data()
        time.sleep(20)