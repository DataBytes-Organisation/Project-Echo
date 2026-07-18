# test_request.py
import requests, sys

API = "http://127.0.0.1:8000/predict" if len(sys.argv) < 2 else sys.argv[1]
AUDIO = "your_audio.wav" if len(sys.argv) < 3 else sys.argv[2]

with open(AUDIO, "rb") as f:
    files = {"file": ("audio.wav", f, "audio/wav")}
    resp = requests.post(API, files=files, timeout=30)
    print(resp.status_code, resp.text)
