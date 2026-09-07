"""Redis query cache for insights (logical DB 2, prefix insights:)."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

from redis import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

INSIGHTS_PREFIX = "insights:"

_client: Optional[Redis] = None


def _ttl_seconds() -> int:
    try:
        return int(os.getenv("CACHE_TTL_SECONDS", "60"))
    except (TypeError, ValueError):
        return 60


def get_client() -> Optional[Redis]:
    global _client
    if _client is None:
        _client = Redis(
            host=os.getenv("REDIS_HOST", "echo-redis"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_CACHE_DB", "2")),
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
    return _client


def _normalize_bound(value: Optional[str]) -> str:
    if not value:
        return "all"
    text = value.strip()
    if not text:
        return "all"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.isoformat()
    except ValueError:
        return text


def insights_overview_key(start: Optional[str] = None, end: Optional[str] = None) -> str:
    return f"{INSIGHTS_PREFIX}overview:{_normalize_bound(start)}:{_normalize_bound(end)}"


def insights_species_key(limit: int) -> str:
    return f"{INSIGHTS_PREFIX}species:{limit}"


def get_json(key: str) -> Optional[Any]:
    client = get_client()
    if client is None:
        return None
    try:
        raw = client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except (RedisError, json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Redis cache get failed for %s", key, exc_info=True)
        return None


def set_json(key: str, data: Any, ttl: Optional[int] = None) -> None:
    client = get_client()
    if client is None:
        return
    try:
        client.setex(key, ttl if ttl is not None else _ttl_seconds(), json.dumps(data, default=str))
    except (RedisError, TypeError, ValueError):
        logger.warning("Redis cache set failed for %s", key, exc_info=True)


def invalidate_insights() -> None:
    client = get_client()
    if client is None:
        return
    try:
        keys = list(client.scan_iter(match=f"{INSIGHTS_PREFIX}*", count=100))
        if keys:
            client.delete(*keys)
    except RedisError:
        logger.warning("Redis cache invalidate failed", exc_info=True)
