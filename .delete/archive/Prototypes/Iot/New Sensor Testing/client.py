import os
import sounddevice as sd
import numpy as np
import paho.mqtt.client as mqtt
import time
import wave
import psutil
import json
import base64

# --- Added for DHT11 + BH1750 ---
try:
    import board
    import adafruit_dht
    DHT_AVAILABLE = True
except Exception:
    board = None
    adafruit_dht = None
    DHT_AVAILABLE = False

try:
    from smbus2 import SMBus
    I2C_AVAILABLE = True
except Exception:
    SMBus = None
    I2C_AVAILABLE = False
# -------------------

AUDIO_DIR = "audioLocal"
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "iot/data/test"

# Sensor config 
DHT_SENSOR = adafruit_dht.DHT11 if DHT_AVAILABLE else None
DHT_PIN = 4  
try:
    DHT_DEVICE = DHT_SENSOR(getattr(board, f"D{DHT_PIN}"), use_pulseio=False) if DHT_AVAILABLE else None
except Exception:
    DHT_DEVICE = None
    DHT_AVAILABLE = False

I2C_BUS = 1
BH1750_ADDRS = (0x23, 0x5C)
BH1750_ONE_TIME_HIRES = 0x20
#---------------

os.makedirs(AUDIO_DIR, exist_ok=True)

sd.default.device = (2, None)

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
        "uptime": psutil.boot_time()
    }

# --- Added sensor read helpers ---
def read_dht11():
    if not DHT_AVAILABLE or DHT_DEVICE is None:
        return {"temp_c": None, "humidity": None}

    try:
        temperature = DHT_DEVICE.temperature
        humidity = DHT_DEVICE.humidity
    except RuntimeError:
        return {"temp_c": None, "humidity": None}
    except Exception:
        return {"temp_c": None, "humidity": None}

    return {
        "temp_c": float(temperature) if temperature is not None else None,
        "humidity": float(humidity) if humidity is not None else None,
    }


def read_bh1750_lux():
    if not I2C_AVAILABLE:
        return None, None

    last_error = None
    for addr in BH1750_ADDRS:
        try:
            with SMBus(I2C_BUS) as bus:
                bus.write_byte(addr, BH1750_ONE_TIME_HIRES)
                time.sleep(0.18)
                data = bus.read_i2c_block_data(addr, 0x00, 2)
                raw = (data[0] << 8) | data[1]
                lux = raw / 1.2
                return float(lux), f"0x{addr:02x}"
        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        print(f"BH1750 read failed: {last_error}")

    return None, None
# ----------------

def send_data():
    audio_file = record_audio()

    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    health_data = get_health_report()

    # --- read DHT11 + BH1750 and include as env_data ---
    env_data = read_dht11()
    lux, addr = read_bh1750_lux()
    env_data["lux"] = lux
    env_data["bh1750_addr"] = addr

    audio_b64 = base64.b64encode(audio_bytes).decode('ascii')

    # ---------------

    payload = {
        "health_data": health_data,
        "env_data": env_data,
        "audio_file": audio_b64
    }

    json_payload = json.dumps(payload)

    client.on_publish = on_publish
    client.publish(topic, json_payload, qos=1)
    print(f"Published json + audio to {topic}: {audio_file} | env_data={env_data}")

if __name__ == "__main__":
    while True:
        send_data()
        time.sleep(20)
