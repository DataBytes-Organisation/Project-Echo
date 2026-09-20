"""Synthetic, read-only S3 responses for the Docker CI startup check only.

This module is mounted by docker-compose.ci.yml, never copied into production
images. It exercises the real R2 storage code without a live bucket or secrets.
"""

import argparse
import io
import os
from pathlib import Path
import runpy
import sys
import wave
from unittest.mock import patch

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber

from cloudflare_r2 import R2Config, R2Storage


CI_ENV = {
    "ECHO_R2_CI_FIXTURE": "1",
    "R2_ACCOUNT_ID": "ci-fixture",
    "R2_BUCKET_NAME": "echo-ci-fixture",
    "R2_ACCESS_KEY_ID": "ci-only-not-a-real-access-key",
    "R2_SECRET_ACCESS_KEY": "ci-only-not-a-real-secret",
    "R2_DATASET_PREFIX": "",
}
SPECIES = "Gymnorhina tibicen"
OBJECT_KEY = f"{SPECIES}/ci-sample.wav"


def sample_audio():
    """Return one second of valid mono 16 kHz PCM audio, generated in memory."""
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16000)
    return output.getvalue()


def validate_environment():
    if any(os.environ.get(name) != value for name, value in CI_ENV.items()):
        raise RuntimeError(
            "R2 CI fixture requires the explicit synthetic CI configuration; "
            "do not use live credentials with this launcher"
        )


class FixtureClient:
    """Use SDK-validated responses, with no storage requests sent over HTTP."""

    def __init__(self, config):
        validate_environment()
        if config != R2Config.from_env():
            raise ValueError("R2 CI fixture only accepts the CI bucket configuration")
        self.bucket = config.bucket_name
        self.audio = sample_audio()
        self.client = boto3.client(
            "s3",
            endpoint_url="http://127.0.0.1:9",
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            region_name="auto",
        )

    def _respond(self, method, response, expected, request):
        with Stubber(self.client) as stubber:
            stubber.add_response(method, response, expected)
            result = getattr(self.client, method)(**request)
            stubber.assert_no_pending_responses()
            return result

    def head_bucket(self, **request):
        return self._respond("head_bucket", {}, {"Bucket": self.bucket}, request)

    def list_objects_v2(self, **request):
        prefix = request.get("Prefix", "")
        objects = (
            [{"Key": OBJECT_KEY, "Size": len(self.audio)}]
            if OBJECT_KEY.startswith(prefix) else []
        )
        return self._respond(
            "list_objects_v2",
            {"Contents": objects, "IsTruncated": False, "KeyCount": len(objects)},
            {"Bucket": self.bucket, "Prefix": prefix},
            request,
        )

    def get_object(self, **request):
        return self._respond(
            "get_object",
            {
                "Body": StreamingBody(io.BytesIO(self.audio), len(self.audio)),
                "ContentLength": len(self.audio),
                "ContentType": "audio/wav",
            },
            {"Bucket": self.bucket, "Key": OBJECT_KEY},
            request,
        )


def verify_storage():
    storage = R2Storage(R2Config.from_env())
    storage.check_connection()
    if storage.group_objects_by_species() != {SPECIES: [OBJECT_KEY]}:
        raise RuntimeError("R2 CI species discovery failed")
    if storage.download_bytes(OBJECT_KEY) != sample_audio():
        raise RuntimeError("R2 CI audio download integrity failed")
    print(
        "R2 CI fixture: PASS (synthetic bucket/list/download; no live R2 access)",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=("engine", "simulator", "check"))
    service = parser.parse_args().service
    validate_environment()
    with patch.object(R2Storage, "_create_client", staticmethod(FixtureClient)):
        verify_storage()
        if service == "check":
            return
        target = Path("/app") / {
            "engine": "echo_engine.py",
            "simulator": "system_manager.py",
        }[service]
        sys.path.insert(0, str(target.parent))
        # Execute the real application; exceptions still fail the CI container.
        runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
