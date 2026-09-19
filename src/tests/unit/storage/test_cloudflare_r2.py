import io
import os
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


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

    def close(self):
        pass


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


class R2PrototypeTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeR2Client()
        self.config = R2Config("account", "bucket", "key", "secret", "prototype")
        self.storage = R2Storage(self.config, self.client)

    def test_config_requires_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "R2_ACCOUNT_ID"):
                R2Config.from_env()

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


class R2ProductionStorageTests(unittest.TestCase):
    def setUp(self):
        self.config = R2Config("account", "bucket", "test-key", "test-secret")
        self.client = MagicMock()
        self.storage = R2Storage(self.config, self.client)

    def test_empty_prefix_is_preserved_from_environment(self):
        settings = {
            "R2_ACCOUNT_ID": "account", "R2_BUCKET_NAME": "bucket",
            "R2_ACCESS_KEY_ID": "test-key", "R2_SECRET_ACCESS_KEY": "test-secret",
        }
        with patch.dict(os.environ, settings, clear=True):
            self.assertEqual(R2Config.from_env().dataset_prefix, "")
            os.environ["R2_DATASET_PREFIX"] = ""
            self.assertEqual(R2Config.from_env().dataset_prefix, "")
            os.environ["R2_DATASET_PREFIX"] = " /dataset/ "
            self.assertEqual(R2Config.from_env().dataset_prefix, "dataset")

    def test_credentials_are_not_in_config_repr(self):
        self.assertNotIn("test-key", repr(self.config))
        self.assertNotIn("test-secret", repr(self.config))

    def test_root_layout_pagination_and_folder_markers(self):
        self.client.list_objects_v2.side_effect = [
            {
                "Contents": [
                    {"Key": "Species B/two.wav", "Size": 100},
                    {"Key": "Species A/one.wav", "Size": 100},
                    {"Key": "Empty/", "Size": 0},
                    {"Key": "Empty/zero.wav", "Size": 0},
                    {"Key": "root-file.wav", "Size": 100},
                ],
                "IsTruncated": True, "NextContinuationToken": "next",
            },
            {"Contents": [{"Key": "Species A/three.wav", "Size": 100}]},
        ]
        self.assertEqual(self.storage.group_objects_by_species(), {
            "Species A": ["Species A/one.wav", "Species A/three.wav"],
            "Species B": ["Species B/two.wav"],
        })
        self.assertEqual(self.client.list_objects_v2.call_args_list[0].kwargs,
                         {"Bucket": "bucket", "Prefix": ""})
        self.assertEqual(self.client.list_objects_v2.call_args.kwargs,
                         {"Bucket": "bucket", "Prefix": "", "ContinuationToken": "next"})

    def test_empty_bucket_returns_no_species(self):
        self.client.list_objects_v2.return_value = {}
        self.assertEqual(self.storage.list_species(), [])

    def test_incomplete_pagination_fails_explicitly(self):
        self.client.list_objects_v2.return_value = {"IsTruncated": True}
        with self.assertRaisesRegex(RuntimeError, "continuation token"):
            self.storage.list_species()

    def test_download_preserves_bytes_and_closes_stream(self):
        body = io.BytesIO(b"RIFF-unchanged-audio")
        self.client.get_object.return_value = {"Body": body}
        self.assertEqual(self.storage.download_bytes("Species A/one.wav"),
                         b"RIFF-unchanged-audio")
        self.assertTrue(body.closed)
        self.client.get_object.assert_called_once_with(Bucket="bucket", Key="Species A/one.wav")

    def test_failed_or_empty_download_closes_stream(self):
        body = MagicMock()
        body.read.side_effect = OSError("interrupted")
        self.client.get_object.return_value = {"Body": body}
        with self.assertRaisesRegex(OSError, "interrupted"):
            self.storage.download_bytes("Species A/one.wav")
        body.close.assert_called_once()
        empty = io.BytesIO()
        self.client.get_object.return_value = {"Body": empty}
        with self.assertRaisesRegex(RuntimeError, "empty"):
            self.storage.download_bytes("Species A/one.wav")
        self.assertTrue(empty.closed)

    def test_sdk_uses_r2_endpoint_and_explicit_credentials(self):
        with patch("boto3.client") as create_client:
            R2Storage(self.config)
        args = create_client.call_args
        self.assertEqual(args.args, ("s3",))
        self.assertEqual(args.kwargs["endpoint_url"], "https://account.r2.cloudflarestorage.com")
        self.assertEqual(args.kwargs["region_name"], "auto")
        self.assertEqual(args.kwargs["aws_access_key_id"], "test-key")
        self.assertEqual(args.kwargs["aws_secret_access_key"], "test-secret")
        self.assertEqual(args.kwargs["config"].connect_timeout, 10)
        self.assertEqual(args.kwargs["config"].read_timeout, 60)


if __name__ == "__main__":
    unittest.main()
