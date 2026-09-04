import time
import socket
from fastapi import APIRouter
from app.database import client
from app.queue import redis_conn
import os

router = APIRouter(prefix="/health", tags=["health"])

def check_socket(host, port, timeout=2.0):
    try:
        start = time.perf_counter()
        with socket.create_connection((host, port), timeout=timeout):
            return {"status": "UP", "latency_ms": round((time.perf_counter() - start) * 1000, 2)}
    except Exception:
        return {"status": "DOWN", "latency_ms": None}

@router.get("/dependencies")
def health_dependencies():
    """
    Returns the health and latency of backend dependencies: MongoDB, Redis, HiveMQ.
    """
    results = {}
    
    # Check MongoDB
    try:
        start = time.perf_counter()
        client.admin.command('ping')
        results["mongodb"] = {"status": "UP", "latency_ms": round((time.perf_counter() - start) * 1000, 2)}
    except Exception:
        results["mongodb"] = {"status": "DOWN", "latency_ms": None}
        
    # Check Redis
    try:
        start = time.perf_counter()
        redis_conn.ping()
        results["redis"] = {"status": "UP", "latency_ms": round((time.perf_counter() - start) * 1000, 2)}
    except Exception:
        results["redis"] = {"status": "DOWN", "latency_ms": None}
        
    # Check HiveMQ (MQTT)
    mqtt_host = os.getenv("MQTT_HOST", "ts-mqtt-server-cont")
    mqtt_port = int(os.getenv("MQTT_PORT", 1883))
    results["hivemq"] = check_socket(mqtt_host, mqtt_port)
    
    # Determine overall status
    overall_status = "UP"
    down_count = sum(1 for svc in results.values() if svc["status"] == "DOWN")
    if down_count == len(results):
        overall_status = "DOWN"
    elif down_count > 0:
        overall_status = "DEGRADED"
        
    return {
        "status": overall_status,
        "dependencies": results
    }
