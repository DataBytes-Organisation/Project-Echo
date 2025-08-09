import psutil
import sounddevice as sd
import wave
import time
import requests

SERVER_URL = "http://192.168.1.110:5000/upload"

def get_health_report():
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
        "uptime": psutil.boot_time()
    }

def record_audio(filename="audio.wav", duration=5, samplerate=44100):
    print("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    print("Recording complete:", filename)
    return filename

def send_data():
    health_data = get_health_report()
    audio_file = record_audio()

    files = {
        "audio": open(audio_file, "rb")
    }
    data = health_data

    response = requests.post(SERVER_URL, data=data, files=files)
    print("Server response:", response.text)

if __name__ == "__main__":
    send_data()
