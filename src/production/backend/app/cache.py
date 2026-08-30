import json
import os
from typing import Any, Optional

import redis

REDIS_HOST = os.getenv("REDIS_HOST", "echo-redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
)

DETECTIONS_CACHE_PREFIX = "detections:list:"
DETECTIONS_CACHE_TTL_SECONDS = 60


def get_cached_json(key: str) -> Optional[Any]:
    value = redis_client.get(key)

    if value is None:
        print(f"Detection cache MISS: {key}")
        return None

    print(f"Detection cache HIT: {key}")
    return json.loads(value)


def set_cached_json(
    key: str,
    value: Any,
    ttl: int = DETECTIONS_CACHE_TTL_SECONDS,
):
    redis_client.setex(
        key,
        ttl,
        json.dumps(value, default=str),
    )


def invalidate_detection_cache():
    deleted = 0

    for key in redis_client.scan_iter(f"{DETECTIONS_CACHE_PREFIX}*"):
        redis_client.delete(key)
        deleted += 1

    print(f"Detection cache invalidated: {deleted} key(s)")