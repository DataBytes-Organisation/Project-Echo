import os 
import psutil
import sounddevice as sd
import numpy as np
import librosa
import cv2
import json
import wave
import time
import requests
import serial
from gps3 import gps3
from tflite_runtime.interpreter import Interpreter

SERVER_URL = "http://192.168.1.122:5000/upload"
SAMPLE_RATE = 16000
DURATION = 5
N_MFCC = 40
MODEL_PATH = "models/checkpoint_MobileNetV3-Large.tflite" 
CLASS_PATH = "models/class_names_MobileNetV3-Large.json"
SAVE_DIR = "audioLocal"
os.makedirs(SAVE_DIR, exist_ok=True)

sd.default.device = (2, None)

print("Loading Model......")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']

with open(CLASS_PATH, "r") as f:
    ID_TO_SPECIES = json.load(f)

def get_health_report():
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
        "uptime": psutil.boot_time()
    }

# sub-optimal, convert to using gps3 and gpsd
def get_gps_location(port="/dev/ttyACM0",baudrate=9600,timeout=1):
    try:
        with serial.Serial(port, baudrate, timeout=timeout) as ser:
            while True:
                print("1")
                line = ser.readline().decode("ascii", errors="replace").strip()
                if line.startswith("$GPGGA") or line.startswith("$GPRMC"):
                    print("2")
                    parts = line.split(",")
                    if len(parts) > 5 and parts[2] and parts[4]:
                        lat = convert_to_decimal(parts[2], parts[3])
                        lon = convert_to_decimal(parts[4], parts[5])
                        print("lat: ", lat)
                        return {"latitude": lat, "longitude": lon}

    except Exception as e:
        print("GPS error: ", e)
    
    return  {"latitude": None, "longitude": None}


#Check audio recording duration and quality, method for sending .wav file to server
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
    return filepath, np.squeeze(audio_data)

# Is processing neccessary? How does the model expect input?
def preprocess_audio(audio, sr=SAMPLE_RATE):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=224, fmax=sr//2)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_norm = 255 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    mel_norm = mel_norm.astype(np.uint8)
    mel_resized = cv2.resize(mel_norm, (224, 224))
    mel_rgb = np.stack([mel_resized]*3, axis=-1)
    mel_rgb = np.expand_dims(mel_rgb, axis=0)

    return mel_rgb.astype(np.float32)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# Does this prediction help the actual model in any way?
def predict_species(audio):
    input_data = preprocess_audio(audio)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)[0]
    
    probs = softmax(output_data)
    
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    
    species = ID_TO_SPECIES[class_id]
    return species, confidence

def cleanup_files(max_files=100):
    files = sorted([os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR)], key=os.path.getmtime)
    if len(files) > max_files:
        for f in files[:-max_files]:
            os.remove(f)
            print(f"Deleted old file: {f}")

def send_data():
    filepath, audio = record_audio()

    species, confidence = predict_species(audio)
    print(f"Predicted: {species} ({confidence:.2f})")

    cleanup_files(max_files=100)

    health_data = get_health_report()
    # gps_data = get_gps_location()

    payload = {
        **health_data,
        # **gps_data,
        "species": species,
        "confidence": confidence
    }

    response = requests.post(SERVER_URL, json=payload)

if __name__ == "__main__":
    while True:
        send_data()
