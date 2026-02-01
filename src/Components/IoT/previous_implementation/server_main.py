from flask import Flask, request
import os
from datetime import datetime

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    
    if data:
        cpu = data.get("cpu")
        ram = data.get("ram")
        disk = data.get("disk")
        uptime = data.get("uptime")
        # latitude = data.get("latitude")
        # longtude = data.get("longitude")
        species = data.get("species")
        confidence = data.get("confidence")
        
        print(f"CPU: {cpu}%, RAM: {ram}%, Disk: {disk}%, Uptime: {uptime}")
        # print(f"Location: {latitude}, {longitude}")
        print(f"Species: {species}, Confidence: {confidence}")
    
    return "Data received successfully!", 200
        


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
