import logging
import socket
import time
from typing import Any, Dict

from fastapi import APIRouter
from app.config import settings
from app.database import client, Userclient
from app.queue import redis_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


def check_socket(host: str, port: int, timeout: float = 2.0) -> Dict[str, Any]:
    """
    Performs a bounded TCP socket connection probe.
    """
    start = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            latency = round((time.perf_counter() - start) * 1000, 2)
            return {"status": "UP", "latency_ms": latency}
    except Exception as exc:
        logger.debug("Socket probe to %s:%s failed: %s", host, port, exc)
        return {"status": "DOWN", "latency_ms": None, "error": str(exc)}


@router.get("/dependencies")
def health_dependencies():
    """
    Returns bounded health status and round-trip latency for all backend dependencies:
    - MongoDB (EchoNet primary database and User database)
    - Redis (Asynchronous job queue)
    - HiveMQ / MQTT broker
    """
    results = {}
    timeout_ms = settings.mongo_timeout_ms

    # 1. Check Primary MongoDB (EchoNet)
    start_mongo = time.perf_counter()
    try:
        client.admin.command("ping", maxTimeMS=timeout_ms)
        results["mongodb_primary"] = {
            "status": "UP",
            "latency_ms": round((time.perf_counter() - start_mongo) * 1000, 2),
        }
    except Exception as exc:
        logger.warning("MongoDB primary health check failed: %s", exc)
        results["mongodb_primary"] = {"status": "DOWN", "latency_ms": None, "error": str(exc)}

    # 2. Check User MongoDB (UserSample)
    start_user_mongo = time.perf_counter()
    try:
        Userclient.admin.command("ping", maxTimeMS=timeout_ms)
        results["mongodb_user"] = {
            "status": "UP",
            "latency_ms": round((time.perf_counter() - start_user_mongo) * 1000, 2),
        }
    except Exception as exc:
        logger.warning("MongoDB user DB health check failed: %s", exc)
        results["mongodb_user"] = {"status": "DOWN", "latency_ms": None, "error": str(exc)}

    # 3. Check Redis (Queue)
    start_redis = time.perf_counter()
    try:
        # redis_conn has socket_timeout and socket_connect_timeout explicitly configured
        redis_conn.ping()
        results["redis"] = {
            "status": "UP",
            "latency_ms": round((time.perf_counter() - start_redis) * 1000, 2),
        }
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        results["redis"] = {"status": "DOWN", "latency_ms": None, "error": str(exc)}

    # 4. Check HiveMQ / MQTT Broker
    results["hivemq"] = check_socket(
        settings.mqtt_host,
        settings.mqtt_port,
        timeout=settings.redis_connect_timeout,
    )

    # Determine overall status
    down_count = sum(1 for svc in results.values() if svc["status"] == "DOWN")
    if down_count == len(results):
        overall_status = "DOWN"
    elif down_count > 0:
        overall_status = "DEGRADED"
    else:
        overall_status = "UP"

    return {
        "status": overall_status,
        "dependencies": results,
    }
