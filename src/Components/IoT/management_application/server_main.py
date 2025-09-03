from flask import Flask, request
import os
from datetime import datetime

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    # Read health data
    cpu = request.form.get("cpu")
    ram = request.form.get("ram")
    disk = request.form.get("disk")
    uptime = request.form.get("uptime")

    print(f"CPU: {cpu}%, RAM: {ram}%, Disk: {disk}%, Uptime: {uptime}")

    # Save audio file
    audio = request.files.get("audio")
    if audio:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        audio.save(filepath)
        print(f"Audio saved at: {filepath}")

    return "Data received successfully!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
