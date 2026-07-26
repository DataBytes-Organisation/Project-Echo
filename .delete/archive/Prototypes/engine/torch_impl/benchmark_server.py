import paho.mqtt.client as mqtt
import json
import time
import base64
import numpy as np
import concurrent.futures

# --- CONFIGURATION ---
BROKER = "localhost"             # your MQTT broker host
PORT = 1883                      # your MQTT broker port
TOPIC = "echo/audio"             # must match engine config['MQTT_PUBLISH_URL']
NUM_MESSAGES = 500               # total messages to send
CONCURRENCY = 10                 # parallel senders
AUDIO_DURATION_SEC = 5           # fake audio duration in seconds
SAMPLE_RATE = 48000

# --- HELPER FUNCTIONS ---
def base64_encode(audio_bytes):
    return base64.b64encode(audio_bytes).decode('utf-8')

def generate_dummy_audio_event():
    """Create a fake audio event in the format Echo Engine expects."""
    fake_audio = (np.random.rand(SAMPLE_RATE * AUDIO_DURATION_SEC).astype(np.float32).tobytes())
    event = {
        "audioClip": base64_encode(fake_audio),
        "audioFile": "Recording_Mode",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sensorId": "TEST",
        "microphoneLLA": [0,0,0],
        "animalEstLLA": [0,0,0],
        "animalTrueLLA": [0,0,0],
        "animalLLAUncertainty": [0,0,0]
    }
    return json.dumps(event)

def send_message():
    """Send a single message and measure time."""
    client = mqtt.Client()
    try:
        client.connect(BROKER, PORT)
        client.loop_start()
        event = generate_dummy_audio_event()
        start_time = time.time()
        client.publish(TOPIC, payload=event)
        latency = (time.time() - start_time) * 1000  # ms
        client.loop_stop()
        client.disconnect()
        return latency, True
    except Exception as e:
        print(f"Send failed: {e}")
        return 0, False

# --- BENCHMARK FUNCTION ---
def run_benchmark():
    print(f"--- MQTT BENCHMARK START ---")
    print(f"Broker: {BROKER}:{PORT} | Topic: {TOPIC}")
    print(f"Messages: {NUM_MESSAGES} | Concurrency: {CONCURRENCY}")

    latencies = []
    success_count = 0
    start_benchmark = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(send_message) for _ in range(NUM_MESSAGES)]
        for future in concurrent.futures.as_completed(futures):
            latency, success = future.result()
            if success:
                latencies.append(latency)
                success_count += 1

    total_time = time.time() - start_benchmark

    if latencies:
        avg_latency = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p90 = np.percentile(latencies, 90)
        p99 = np.percentile(latencies, 99)
        throughput = success_count / total_time
    else:
        avg_latency = p50 = p90 = p99 = throughput = 0.0

    print("\n" + "="*30)
    print(f"MQTT BENCHMARK REPORT")
    print("="*30)
    print(f"Total Time     : {total_time:.2f} seconds")
    print(f"Successful Msg : {success_count}/{NUM_MESSAGES}")
    print("-" * 30)
    print(f"Throughput     : {throughput:.2f} msg/sec")
    print("-" * 30)
    print(f"Latency (Avg)  : {avg_latency:.2f} ms")
    print(f"Latency (P50)  : {p50:.2f} ms (Median)")
    print(f"Latency (P90)  : {p90:.2f} ms")
    print(f"Latency (P99)  : {p99:.2f} ms")
    print("="*30)

if __name__ == "__main__":
    run_benchmark()
