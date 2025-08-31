import os
import pytest
import asyncio
from httpx import AsyncClient

# These tests assume the API is running locally at 127.0.0.1:9000
BASE = os.getenv("API_BASE", "http://127.0.0.1:9000")
SAMPLE = os.getenv("AUDIO_SAMPLE", "/Users/tenniele/Documents/Project A/Project-Echo/Audio_testAPI.mp3")

pytestmark = pytest.mark.asyncio

async def _skip_if_no_sample():
    if not os.path.exists(SAMPLE):
        pytest.skip(f"Sample file not found: {SAMPLE}")

async def test_upload_ok():
    await _skip_if_no_sample()
    async with AsyncClient(base_url=BASE) as ac:
        with open(SAMPLE, "rb") as f:
            files = {"file": (os.path.basename(SAMPLE), f, "audio/mpeg")}
            data = {"user_id": "ci-user"}
            resp = await ac.post("/api/audio/upload", files=files, data=data)
            assert resp.status_code == 200, resp.text
            payload = resp.json()
            assert "upload_id" in payload
            assert payload["message"] == "Upload successful"

async def test_predict_ok():
    await _skip_if_no_sample()
    async with AsyncClient(base_url=BASE) as ac:
        with open(SAMPLE, "rb") as f:
            files = {"audio": (os.path.basename(SAMPLE), f, "audio/mpeg")}
            data = {"user_id": "ci-user"}
            resp = await ac.post("/predict", files=files, data=data)
            assert resp.status_code == 200, resp.text
            payload = resp.json()
            assert "species" in payload and "confidence" in payload
