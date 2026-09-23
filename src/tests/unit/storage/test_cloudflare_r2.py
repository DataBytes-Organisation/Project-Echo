import io
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


STORE_DIR = Path(__file__).resolve().parents[3] / "production" / "infrastructure" / "store"
sys.path.insert(0, str(STORE_DIR))

from cloudflare_r2 import (
    R2Config,
    R2EnginePrototype,
    R2SimulatorPrototype,
    R2Storage,
)


class FakeBody:
    def __init__(self, value):
        self.value = value

    def read(self):
        return self.value


class FakeR2Client:
    def __init__(self):
        self.requests = []
        self.upload = None

    def head_bucket(self, **kwargs):
        self.head_request = kwargs

    def list_objects_v2(self, **kwargs):
        self.requests.append(kwargs)
        if "ContinuationToken" not in kwargs:
            return {
                "Contents": [
                    {"Key": "prototype/Species B/two.wav"},
                    {"Key": "prototype/Species A/one.wav"},
                    {"Key": "prototype/folder-only/"},
                ],
                "IsTruncated": True,
                "NextContinuationToken": "next",
            }
        return {
            "Contents": [{"Key": "prototype/Species A/nested/three.mp3"}],
            "IsTruncated": False,
        }

    def get_object(self, **kwargs):
        self.get_request = kwargs
        return {"Body": FakeBody(b"RIFF-audio")}

    def upload_fileobj(self, source, bucket, key, **kwargs):
        self.upload = (source.read(), bucket, key, kwargs)

    def delete_object(self, **kwargs):
        self.delete_request = kwargs


class R2PrototypeTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeR2Client()
        self.config = R2Config(
            account_id="account",
            bucket_name="bucket",
            access_key_id="key",
            secret_access_key="secret",
            endpoint_url="https://account.r2.cloudflarestorage.com",
        )
        self.storage = R2Storage(self.config, self.client)

    def test_config_requires_credentials(self):
        app_settings = SimpleNamespace(
            r2_account_id=None,
            r2_bucket_name=None,
            r2_access_key_id=None,
            r2_secret_access_key=None,
            r2_endpoint_url=None,
            r2_dataset_prefix="prototype",
        )

        with self.assertRaisesRegex(ValueError, "R2_ACCOUNT_ID"):
            R2Config.from_settings(app_settings)

    def test_config_reads_validated_backend_settings(self):
        app_settings = SimpleNamespace(
            r2_account_id="account-123",
            r2_bucket_name="echo-bucket",
            r2_access_key_id="access-key",
            r2_secret_access_key="secret-key",
            r2_endpoint_url="https://account-123.r2.cloudflarestorage.com",
            r2_dataset_prefix="dataset",
        )

        config = R2Config.from_settings(app_settings)

        self.assertEqual(config.account_id, "account-123")
        self.assertEqual(config.bucket_name, "echo-bucket")
        self.assertEqual(config.access_key_id, "access-key")
        self.assertEqual(config.secret_access_key, "secret-key")
        self.assertEqual(
            config.endpoint_url,
            "https://account-123.r2.cloudflarestorage.com",
        )
        self.assertEqual(config.dataset_prefix, "dataset")

    def test_connection_uses_configured_bucket(self):
        self.storage.check_connection()
        self.assertEqual(self.client.head_request, {"Bucket": "bucket"})

    def test_paginated_grouping_matches_species_layout(self):
        self.assertEqual(
            self.storage.group_objects_by_species(),
            {
                "Species A": [
                    "prototype/Species A/nested/three.mp3",
                    "prototype/Species A/one.wav",
                ],
                "Species B": ["prototype/Species B/two.wav"],
            },
        )
        self.assertEqual(self.client.requests[1]["ContinuationToken"], "next")

    def test_engine_loads_sorted_species(self):
        engine = R2EnginePrototype(self.storage)
        self.assertEqual(engine.load_species_list(), ["Species A", "Species B"])

    def test_simulator_downloads_random_species_audio(self):
        simulator = R2SimulatorPrototype(
            self.storage, random_source=random.Random(1)
        )
        self.assertEqual(simulator.load_species_list(), ["Species A", "Species B"])
        sample = simulator.download_random_audio("Species B")
        self.assertEqual(sample.file_name, "two.wav")
        self.assertEqual(sample.audio_bytes, b"RIFF-audio")

    def test_simulator_rejects_unknown_species(self):
        simulator = R2SimulatorPrototype(self.storage)
        simulator.load_species_list()
        with self.assertRaisesRegex(KeyError, "Unknown"):
            simulator.download_random_audio("Unknown")

    def test_upload_preserves_audio_content_type(self):
        self.storage.upload_file(
            io.BytesIO(b"sample"),
            "prototype/Species A/sample.wav",
            "audio/wav",
        )
        self.assertEqual(
            self.client.upload,
            (
                b"sample",
                "bucket",
                "prototype/Species A/sample.wav",
                {"ExtraArgs": {"ContentType": "audio/wav"}},
            ),
        )

    def test_invalid_file_key_is_rejected(self):
        with self.assertRaises(ValueError):
            self.storage.download_bytes("prototype/Species A/")

    def test_delete_object_uses_configured_bucket(self):
        self.storage.delete_object(
            "audio_uploads/sample.wav"
        )

        self.assertEqual(
            self.client.delete_request,
            {
                "Bucket": "bucket",
                "Key": "audio_uploads/sample.wav",
            },
        )


if __name__ == "__main__":
    unittest.main()
